#!/usr/bin/env python3
"""
argus_export_onnx.py — stock YOLOv11n -> ONNX, for TensorRT engine build.

Stage 1 of the deployment path:

    yolo11n.pt  ->  ONNX (opset 17, static 640 or 1280)  ->  FP16 or INT8 engine

PURPOSE
  Produce an architecture-neutral ONNX that trtexec can consume on the Orin
  Nano. Runs on the x86 workstation, never on the Jetson: ultralytics declares
  torch and torchvision as hard dependencies, and installing it into the
  Jetson venv pulls generic PyPI wheels over the JetPack CUDA build. ONNX is
  portable, so there is no reason to export on-device.

  Exports FP32. FP16 or INT8 is selected later, when the engine is built on the
  device. Quantizing here as well would lose precision for no benefit.

  Alongside the ONNX this writes a metadata sidecar (class names, tensor names,
  shapes, dtypes, checksums). The Jetson has no ultralytics installed, so that
  sidecar is the only way the inference script learns the COCO class names
  without hardcoding eighty strings.

USAGE
  # inside the maritime-cv conda env -- not a bare shell
  python scripts/argus_export_onnx.py
  python scripts/argus_export_onnx.py --weights /path/to/yolo11n.pt
  python scripts/argus_export_onnx.py --outdir runs --imgsz 640 --opset 17
  python scripts/argus_export_onnx.py --imgsz 1280

OUTPUTS
  runs/yolo11n_<imgsz>.onnx        FP32, static batch 1, imgsz x imgsz
  runs/yolo11n_<imgsz>_meta.json   class names + tensor signature + SHA256

EXIT CODES
  0  ONNX written and verified
  1  export or verification failed
  2  prerequisite missing (ultralytics) or bad arguments

Needs no CUDA and makes no claim about the torch build: the JetPack version
constraints that govern the Jetson do not apply on x86.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Weight resolution
#
# A stock yolo11n.pt already exists in the sibling detector project. Preferring
# it keeps this artifact on a known weight file rather than a network download
# that could quietly hand us a different digest.
# ---------------------------------------------------------------------------

WEIGHT_CANDIDATES = [
    os.path.expanduser("~/projects/computer-vision/maritime-vision/src/models/yolo11n.pt"),
]


def resolve_weights(explicit):
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"weights not found: {explicit}")
        return explicit, "explicit"
    for path in WEIGHT_CANDIDATES:
        if os.path.isfile(path):
            return path, "local"
    # Bare name: ultralytics downloads stock COCO weights on first use.
    return "yolo11n.pt", "download"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export(args):
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"ultralytics not importable ({e}).\n"
                 "Activate the maritime-cv conda env, or "
                 "'pip install ultralytics' there. Do not do this on the Jetson.")

    src, origin = resolve_weights(args.weights)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"ultralytics {ultralytics.__version__}")
    print(f"weights     {src} ({origin})")

    # Ultralytics writes the .onnx next to the .pt it loaded. Staging the
    # weights into outdir first keeps every artifact inside this repo instead
    # of scattering exports through whatever directory the weights came from.
    staged = os.path.join(args.outdir, "yolo11n.pt")
    if origin == "download":
        staged = src
    elif os.path.abspath(src) != os.path.abspath(staged):
        shutil.copy2(src, staged)
        print(f"staged      {staged}")

    model = YOLO(staged)
    names = {int(k): v for k, v in model.names.items()}
    print(f"classes     {len(names)}")

    # simplify=True pulls onnxslim through ultralytics' AutoUpdate. If that
    # install is unavailable the requirement check raises before any export
    # happens, so fall back rather than losing the run. Whether slimming was
    # actually applied is recorded in the sidecar, not assumed.
    simplify = args.simplify
    try:
        out = run_export(model, args, simplify)
    except Exception as e:
        if not simplify:
            raise
        print(f"\nwarning: export with simplify=True failed ({e})")
        print("warning: retrying with simplify=False\n")
        simplify = False
        out = run_export(model, args, simplify)

    onnx_path = os.path.join(args.outdir, f"yolo11n_{args.imgsz}.onnx")
    if os.path.abspath(out) != os.path.abspath(onnx_path):
        shutil.move(out, onnx_path)

    return onnx_path, names, ultralytics.__version__, simplify


def run_export(model, args, simplify):
    # No half/quantize argument: FP32 is the export default. The device build
    # selects FP16 or INT8. device='cpu' keeps the export deterministic and
    # independent of whatever GPU the workstation happens to have.
    return str(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=simplify,
        dynamic=False,
        device="cpu",
    ))


# ---------------------------------------------------------------------------
# Verification
#
# The inference script needs exact tensor names, shapes, and dtypes. Reading
# them back out of the file is the only way to know them; guessing them from a
# tutorial is how the TRT binding setup silently goes wrong.
# ---------------------------------------------------------------------------


def tensor_signature(onnx_path):
    import onnx
    from onnx import TensorProto

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    dtype_name = {v: k for k, v in TensorProto.DataType.items()}

    def describe(values):
        out = []
        for v in values:
            dims = []
            for d in v.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
            out.append({
                "name": v.name,
                "shape": dims,
                "dtype": dtype_name.get(v.type.tensor_type.elem_type, "UNKNOWN"),
            })
        return out

    initializers = {i.name for i in model.graph.initializer}
    inputs = [v for v in model.graph.input if v.name not in initializers]

    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    return describe(inputs), describe(model.graph.output), opset


def report(inputs, outputs, opset, digest, onnx_path):
    print("\n--- ONNX signature ---")
    print(f"opset       {opset}")
    for t in inputs:
        print(f"input       {t['name']:12s} {t['shape']}  {t['dtype']}")
    for t in outputs:
        print(f"output      {t['name']:12s} {t['shape']}  {t['dtype']}")

    size = os.path.getsize(onnx_path) / 1e6
    print(f"\nfile        {onnx_path}  ({size:.2f} MB)")
    print(f"sha256      {digest}")

    dynamic = [d for t in inputs + outputs for d in t["shape"] if not isinstance(d, int)]
    if dynamic:
        print(f"\nwarning: non-static dimensions present {dynamic}; "
              "the engine build expects fully static shapes")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="",
                   help="path to yolo11n.pt (default: local stock weights, else download)")
    p.add_argument("--outdir", default="runs",
                   help="destination for the ONNX and metadata sidecar")
    p.add_argument("--imgsz", type=int, default=640, choices=(640, 1280),
                   help="static square input, 640 or 1280")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-simplify", dest="simplify", action="store_false",
                   help="skip onnxslim (avoids the AutoUpdate install)")
    args = p.parse_args()

    onnx_path, names, ul_version, simplified = export(args)

    inputs, outputs, opset = tensor_signature(onnx_path)
    digest = sha256(onnx_path)
    report(inputs, outputs, opset, digest, onnx_path)

    expect = [1, 3, args.imgsz, args.imgsz]
    if len(inputs) != 1 or inputs[0]["shape"] != expect:
        got = inputs[0]["shape"] if inputs else None
        sys.exit(f"ONNX input shape {got} does not match {expect}")

    try:
        import onnxslim
        slim_version = onnxslim.__version__
    except ImportError:
        slim_version = None

    import torch

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "produced_on": "x86 workstation (WSL2)",
        "model": "yolo11n stock COCO",
        "onnx_file": os.path.basename(onnx_path),
        "onnx_sha256": digest,
        "opset": opset,
        "imgsz": args.imgsz,
        "precision": "fp32 ONNX; FP16 or INT8 is selected when the engine is built",
        "simplify_requested": simplified,
        "onnxslim_version": slim_version,
        "ultralytics_version": ul_version,
        "torch_version": torch.__version__,
        "inputs": inputs,
        "outputs": outputs,
        "names": names,
    }

    meta_path = os.path.join(args.outdir, f"yolo11n_{args.imgsz}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta        {meta_path}")

    print("\nNext: copy the ONNX and the metadata sidecar to the Jetson, "
          "verify the SHA256 there, then run argus_build_engine.sh.")


if __name__ == "__main__":
    main()
