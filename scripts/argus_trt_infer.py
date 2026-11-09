#!/usr/bin/env python3
"""
argus_trt_infer.py — TensorRT YOLOv11n inference and latency measurement.

Stage 3 of the deployment path:

    yolo11n.pt -> ONNX -> FP16 or INT8 engine (640 or 1280) -> [ this script ]

PURPOSE
  Runs a trtexec-built engine on the Orin Nano, two ways:

    --mode image   one frame in, annotated frame out, detections printed with
                   bounding box heights in source pixels
    --mode bench   synthetic input, warmup then measured iterations, reporting
                   GPU-only and end-to-end latency percentiles separately

  Runs on the Jetson and nowhere else. The engine is compiled for sm_87 and
  will not deserialize on the workstation, and a latency figure from an x86 box
  would not describe this system. The script refuses to execute off-device
  rather than emit a number that cannot be used.

ENVIRONMENT
  Requires the project venv, which was created with --system-site-packages.
  That flag is the only reason 'import tensorrt' resolves. If the import fails,
  activate the venv:

      source .venv/bin/activate

  Do NOT 'pip install tensorrt' -- that pulls an x86-oriented wheel and will
  not fix it. TensorRT and torch here are system/JetPack builds.

USAGE
  python scripts/argus_trt_infer.py --mode bench
  python scripts/argus_trt_infer.py --mode bench --iters 500 --source-res 1920x1080
  python scripts/argus_trt_infer.py --mode image --source runs/rooftop.png
  python scripts/argus_trt_infer.py --mode image --source frame.png --conf 0.10

NOTES
  TensorRT 10 uses the v3 tensor API. The v8-era pattern (a bindings list,
  execute_async_v2, get_binding_shape) is gone; anything written against it
  will not run. Device memory is allocated through torch, which is already
  present and CUDA-functional, so neither pycuda nor cuda-python is needed.

  Preprocessing is letterbox + BGR->RGB + CHW + /255. Ultralytics models use no
  ImageNet mean/std normalisation; adding one is a silent accuracy bug.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

ON_DEVICE_LABEL = "on-device (Jetson Orin Nano)"


def require_cv2():
    if cv2 is None:
        sys.exit("OpenCV (cv2) is not importable.\n"
                 "JetPack ships OpenCV system-wide; the venv must have been "
                 "created with --system-site-packages to see it.")
    return cv2


# ---------------------------------------------------------------------------
# Host guard
#
# Section 6 of the task plan: a latency number carries the machine it was
# measured on, and a workstation figure is a different measurement entirely.
# Rather than trust discipline, refuse to run anywhere but Tegra.
# ---------------------------------------------------------------------------


def assert_on_device():
    machine = platform.machine()
    tegra = os.path.exists("/etc/nv_tegra_release") or os.path.exists("/sys/devices/soc0/family")
    if machine == "aarch64" and tegra:
        return
    sys.exit(
        f"refusing to run: this is not the Jetson (machine={machine}, "
        f"tegra_release={'yes' if tegra else 'no'}).\n"
        "The engine is built for sm_87 and will not deserialize here, and a\n"
        "latency figure measured on the workstation is not a number this\n"
        "project can use. Run this over SSH on the device instead.\n\n"
        "The preprocess/postprocess functions in this module are importable\n"
        "off-device for correctness checks; only execution is blocked."
    )


def power_mode():
    """`nvpmodel -q` collapsed to one line. Every latency figure needs this."""
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    parts = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return " / ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Preprocessing
#
# Pure numpy + cv2, no torch, no tensorrt: importable on any host so the
# geometry and the postprocess can be validated off-device.
# ---------------------------------------------------------------------------


def letterbox(img, size=640, pad=114):
    """
    Resize preserving aspect ratio and centre-pad to size x size.

    Returns the padded image plus the transform needed to map coordinates back:
    scale ratio, and the left/top pad in letterboxed pixels.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))

    out = np.full((size, size, 3), pad, dtype=np.uint8)
    if (nw, nh) != (w, h):
        img = require_cv2().resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = img
    return out, r, dx, dy


def preprocess(img, size, dtype):
    """BGR uint8 HWC -> NCHW tensor in [0, 1]. Division by 255 and nothing else."""
    lb, r, dx, dy = letterbox(img, size)
    x = lb[:, :, ::-1].transpose(2, 0, 1)            # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None].astype(dtype, copy=False), r, dx, dy


# ---------------------------------------------------------------------------
# Postprocessing
#
# The v8/v11 head emits (1, 84, 8400): 4 box coordinates (cx, cy, w, h in
# 640-space) followed by 80 class scores. There is no separate objectness
# channel, so confidence is simply the maximum class score.
#
# NMS is implemented here rather than imported from torchvision: torchvision is
# not installed on the device and must not be, since pulling it from PyPI would
# overwrite the JetPack torch build.
# ---------------------------------------------------------------------------


def nms(boxes, scores, iou_thr):
    """Single-class NMS on xyxy boxes. Returns kept indices, highest score first."""
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def postprocess(out, r, dx, dy, src_shape, conf_thr=0.25, iou_thr=0.45):
    """
    (1, 84, 8400) -> list of detections in source-image coordinates.

    Each detection is (x1, y1, x2, y2, confidence, class_id).
    """
    pred = np.asarray(out, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                                 # (8400, 84)

    boxes, scores = pred[:, :4], pred[:, 4:]
    conf = scores.max(axis=1)
    cls = scores.argmax(axis=1)

    m = conf >= conf_thr
    if not m.any():
        return np.zeros((0, 6), dtype=np.float32)
    boxes, conf, cls = boxes[m], conf[m], cls[m]

    cx, cy, w, h = boxes.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Class-wise: boxes of different classes never suppress each other.
    keep = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        keep.append(idx[nms(xyxy[idx], conf[idx], iou_thr)])
    keep = np.concatenate(keep)
    keep = keep[conf[keep].argsort()[::-1]]

    xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

    # Undo the letterbox: remove padding, then rescale to source resolution.
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dx) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dy) / r
    sh, sw = src_shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, sw - 1)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, sh - 1)

    return np.concatenate([xyxy, conf[:, None], cls[:, None].astype(np.float32)], axis=1)


# ---------------------------------------------------------------------------
# TensorRT 10 engine wrapper
#
# tensorrt and torch are imported here, not at module scope, so that the numpy
# geometry above stays importable on a host with neither.
# ---------------------------------------------------------------------------


class TRTEngine:
    def __init__(self, path):
        try:
            import tensorrt as trt
        except ImportError as e:
            sys.exit(f"import tensorrt failed ({e}).\n"
                     "Activate the project venv -- it was created with\n"
                     "--system-site-packages, which is what makes the system\n"
                     "TensorRT bindings visible. Do not pip install tensorrt.")
        try:
            import torch
        except ImportError as e:
            sys.exit(f"import torch failed ({e}). Activate the project venv.")

        self.trt, self.torch = trt, torch

        if not torch.cuda.is_available():
            sys.exit("torch reports no CUDA device; cannot allocate device buffers.")

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(path, "rb") as f:
            blob = f.read()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            sys.exit(f"failed to deserialize {path}.\n"
                     "An engine is specific to the GPU architecture and the\n"
                     "TensorRT version that built it. Rebuild it on this device\n"
                     "with argus_build_engine.sh.")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            sys.exit("failed to create execution context")

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.zeros(1, dtype=np_dtype)).dtype
            buf = torch.empty(shape, dtype=torch_dtype, device="cuda")
            # v3 API: addresses are bound by tensor name, once, up front.
            self.context.set_tensor_address(name, buf.data_ptr())
            rec = {"name": name, "shape": shape, "np_dtype": np_dtype, "buf": buf}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(rec)
            else:
                self.outputs.append(rec)

        if len(self.inputs) != 1 or len(self.outputs) != 1:
            sys.exit(f"expected 1 input and 1 output, got "
                     f"{len(self.inputs)} and {len(self.outputs)}")

        self.stream = torch.cuda.current_stream()

    @property
    def input_shape(self):
        return self.inputs[0]["shape"]

    @property
    def input_dtype(self):
        return self.inputs[0]["np_dtype"]

    def describe(self):
        lines = []
        for tag, recs in (("input", self.inputs), ("output", self.outputs)):
            for r in recs:
                lines.append(f"{tag:8s} {r['name']:10s} {list(r['shape'])} "
                             f"{np.dtype(r['np_dtype']).name}")
        return lines

    def load_input(self, x):
        """Host NCHW array -> device input buffer."""
        src = self.torch.from_numpy(np.ascontiguousarray(x, dtype=self.input_dtype))
        self.inputs[0]["buf"].copy_(src.to(self.inputs[0]["buf"].dtype))

    def run(self):
        """Execute and synchronise. This is the GPU-only measurement boundary."""
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("execute_async_v3 returned False")
        self.stream.synchronize()

    def fetch_output(self):
        return self.outputs[0]["buf"].detach().cpu().numpy()

    def infer(self, x):
        self.load_input(x)
        self.run()
        return self.fetch_output()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def class_names(meta_path, n=80):
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        names = meta.get("names") or {}
        if names:
            return {int(k): v for k, v in names.items()}
        print(f"warning: {meta_path} has no class names", file=sys.stderr)
    else:
        print(f"warning: metadata sidecar not found ({meta_path}); "
              "labelling detections by class index", file=sys.stderr)
    return {i: f"class_{i}" for i in range(n)}


def header(args, engine, power):
    """Provenance block. Every field here is required to make the numbers mean something."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    shape = engine.input_shape
    res = f"{shape[3]}x{shape[2]}" if len(shape) == 4 else str(list(shape))

    print("=" * 68)
    print(f"argus_trt_infer.py  --mode {args.mode}")
    print("=" * 68)
    print(f"host            {ON_DEVICE_LABEL}")
    print(f"platform        {platform.machine()} / {platform.system()} "
          f"{platform.release()}")
    print(f"utc             {stamp}")
    print(f"engine          {args.engine}")
    print(f"precision       {args.precision} (trtexec build flag)")
    print(f"input res       {res}")
    for line in engine.describe():
        print(f"tensor          {line}")
    print(f"tensorrt        {engine.trt.__version__}")
    print(f"torch           {engine.torch.__version__}")
    if power:
        print(f"power mode      {power}")
    else:
        print("power mode      UNAVAILABLE")
        print("                nvpmodel -q gave nothing. Latency figures below")
        print("                are unusable until a power state is recorded")
        print("                alongside them -- pass --power-mode to state it.")
    print("=" * 68)


def percentiles(samples_ms):
    a = np.asarray(samples_ms, dtype=np.float64)
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    return {"p50": p50, "p90": p90, "p99": p99,
            "mean": a.mean(), "min": a.min(), "max": a.max(), "n": a.size}


def print_latency(title, note, stats):
    print(f"\n{title}")
    print(f"  {note}")
    print(f"  {'p50':>9} {'p90':>9} {'p99':>9} {'mean':>9} "
          f"{'min':>9} {'max':>9} {'n':>6}")
    print(f"  {stats['p50']:>9.3f} {stats['p90']:>9.3f} {stats['p99']:>9.3f} "
          f"{stats['mean']:>9.3f} {stats['min']:>9.3f} {stats['max']:>9.3f} "
          f"{stats['n']:>6d}")
    print("  all values in milliseconds")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_bench(args, engine):
    require_cv2()

    try:
        sw, sh = (int(v) for v in args.source_res.lower().split("x"))
    except ValueError:
        sys.exit(f"--source-res wants WxH, got '{args.source_res}'")

    shape = engine.input_shape
    size = shape[2]
    rng = np.random.default_rng(0)

    # Synthetic frame at capture resolution. Preprocess cost scales with source
    # size, so the end-to-end number is only comparable at a stated resolution.
    frame = rng.integers(0, 256, size=(sh, sw, 3), dtype=np.uint8)
    x, r, dx, dy = preprocess(frame, size, engine.input_dtype)

    print(f"\nsynthetic source  {sw}x{sh} uint8 BGR (seeded, not a real capture)")
    print(f"warmup            {args.warmup} iterations")
    print(f"measured          {args.iters} iterations")

    engine.load_input(x)
    for _ in range(args.warmup):
        engine.run()

    # GPU-only: execute + stream sync. No host copies, no preprocess.
    gpu_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        engine.run()
        gpu_ms.append((time.perf_counter() - t0) * 1e3)

    # End-to-end: preprocess, host->device, execute, device->host, postprocess.
    e2e_ms = []
    dets = 0
    for _ in range(args.iters):
        t0 = time.perf_counter()
        x, r, dx, dy = preprocess(frame, size, engine.input_dtype)
        out = engine.infer(x)
        d = postprocess(out, r, dx, dy, frame.shape, args.conf, args.iou)
        e2e_ms.append((time.perf_counter() - t0) * 1e3)
        dets = len(d)

    print_latency("GPU-only latency",
                  "execute_async_v3 + stream synchronise; excludes all host work",
                  percentiles(gpu_ms))
    print_latency("End-to-end latency",
                  f"letterbox {sw}x{sh}->{size} + H2D + execute + D2H + NMS",
                  percentiles(e2e_ms))

    g, e = percentiles(gpu_ms), percentiles(e2e_ms)
    print(f"\nhost overhead     {e['p50'] - g['p50']:.3f} ms at p50 "
          f"({100 * (e['p50'] - g['p50']) / e['p50']:.1f}% of end-to-end)")
    print(f"throughput        {1e3 / e['p50']:.1f} fps at end-to-end p50")
    print(f"\ndetections on synthetic noise: {dets} "
          "(noise is not a scene; this only proves the path executes)")
    print("\nThese two numbers are different measurements. Reporting the "
          "GPU-only\nfigure as though it were end-to-end would overstate the "
          "system by\nroughly the host overhead shown above.")


def mode_image(args, engine, names):
    cv = require_cv2()
    if not args.source:
        sys.exit("--mode image needs --source <path to png/jpg>")
    if not os.path.isfile(args.source):
        sys.exit(f"source not found: {args.source}")

    frame = cv.imread(args.source, cv.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"could not decode {args.source}")

    size = engine.input_shape[2]
    print(f"\nsource            {args.source}  "
          f"{frame.shape[1]}x{frame.shape[0]}")

    x, r, dx, dy = preprocess(frame, size, engine.input_dtype)
    out = engine.infer(x)
    dets = postprocess(out, r, dx, dy, frame.shape, args.conf, args.iou)

    print(f"threshold         conf {args.conf}  iou {args.iou}")
    print(f"detections        {len(dets)}")

    if len(dets):
        print(f"\n  {'#':>3} {'class':<16} {'conf':>6} "
              f"{'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6} "
              f"{'w_px':>6} {'h_px':>6}")
        for i, (x1, y1, x2, y2, conf, c) in enumerate(dets):
            label = names.get(int(c), f"class_{int(c)}")
            print(f"  {i:>3} {label:<16} {conf:>6.3f} "
                  f"{x1:>6.0f} {y1:>6.0f} {x2:>6.0f} {y2:>6.0f} "
                  f"{x2 - x1:>6.1f} {y2 - y1:>6.1f}")
        # Box height in source pixels is the figure the pixel-cascade argument
        # needs: h_px is what the sensor saw, h_px * r is what survived the
        # resize to the engine spatial size, and that divided by 8 is how many
        # P3 stride-8 cells the object actually occupied.
        heights = dets[:, 3] - dets[:, 1]
        tallest, shortest = heights.max(), heights.min()
        print(f"\n  letterbox scale   {r:.4f}  "
              f"({frame.shape[1]}x{frame.shape[0]} -> {size}x{size})")
        print(f"  tallest box       {tallest:.1f}px source -> "
              f"{tallest * r:.1f}px at {size} -> {tallest * r / 8:.2f} P3 cells")
        print(f"  shortest box      {shortest:.1f}px source -> "
              f"{shortest * r:.1f}px at {size} -> {shortest * r / 8:.2f} P3 cells")
    else:
        print("\n  no detections above threshold")
        print("  A negative result is a result. Do not lower --conf to "
              "manufacture\n  a detection; report that the stock COCO detector "
              "did not fire.")

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.source))[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(args.outdir, f"{stem}_det_{stamp}.png")
    cv.imwrite(dest, annotate(frame.copy(), dets, names))
    print(f"\nannotated         {dest}")


def annotate(img, dets, names):
    cv = require_cv2()
    h = img.shape[0]
    thick = max(1, round(h / 600))
    scale = max(0.4, h / 1400)

    for x1, y1, x2, y2, conf, c in dets:
        c = int(c)
        # Deterministic per-class colour, so the same class looks the same
        # across every frame in a presentation.
        colour = (int(37 * (c + 1) % 256), int(91 * (c + 1) % 256),
                  int(173 * (c + 1) % 256))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv.rectangle(img, p1, p2, colour, thick)

        text = f"{names.get(c, f'class_{c}')} {conf:.2f}"
        (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(th + base, p1[1])
        cv.rectangle(img, (p1[0], ty - th - base), (p1[0] + tw, ty), colour, -1)
        cv.putText(img, text, (p1[0], ty - base),
                   cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                   cv.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["image", "bench"], required=True)
    p.add_argument("--engine", default="runs/yolo11n_fp16_640.engine")
    p.add_argument("--meta", default="runs/yolo11n_640_meta.json",
                   help="export sidecar, read for class names")
    p.add_argument("--source", default="", help="input image for --mode image")
    p.add_argument("--outdir", default="runs")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--source-res", default="1920x1080",
                   help="synthetic source resolution for --mode bench")
    p.add_argument("--precision", default="fp16",
                   help="precision the engine was built with, for the header")
    p.add_argument("--power-mode", default="",
                   help="state the power mode if nvpmodel -q is unavailable")
    args = p.parse_args()

    assert_on_device()

    if not os.path.isfile(args.engine):
        sys.exit(f"engine not found: {args.engine}\n"
                 "Build it on this device with argus_build_engine.sh.")

    engine = TRTEngine(args.engine)
    header(args, engine, args.power_mode or power_mode())

    if args.mode == "bench":
        mode_bench(args, engine)
    else:
        mode_image(args, engine, class_names(args.meta))


if __name__ == "__main__":
    main()
