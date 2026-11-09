# Stock YOLOv11n on the Orin Nano — deployment path demo

## What this is

A stock COCO YOLOv11n detector taken end to end onto the edge device:

```
ultralytics yolo11n.pt  ->  ONNX (opset 17, static 640 or 1280)  ->  TensorRT FP16 or INT8 engine (sm_87)  ->  Python inference
```

No training, no fine-tuning, no maritime weights. The artifact under test is the
*deployment path*, not the detector.

## What it does and does not show

> This demonstrates the deployment path (ONNX -> TensorRT -> on-device
> inference) with stock COCO weights. It is not a maritime detector, and its
> detections on maritime scenes are not a performance claim about the capstone
> system.

Two further boundaries worth stating explicitly:

- Latency figures here are **on-device Jetson figures**. Capture-chain
  experiments (including chroma-sweep detection rates) live in
  [maritime-edge](https://github.com/joejyoung/maritime-edge) and are measured
  on an x86 workstation. They are different measurements and must not be
  presented side by side as though they were comparable.
- Every latency figure is only valid at the power mode recorded next to it.

## Deliverables

| File | Host | Role |
|---|---|---|
| `scripts/argus_export_onnx.py` | x86 workstation | `.pt` -> ONNX at 640 or 1280, plus metadata sidecar |
| `scripts/argus_build_engine.sh` | Orin Nano | FP16 (`trtexec`) or INT8 build, plus provenance log |
| `scripts/argus_build_int8.py` | Orin Nano | entropy calibrator and INT8 engine build |
| `scripts/argus_trt_infer.py` | Orin Nano | inference and latency measurement |
| `scripts/argus_live_demo.py` | Orin Nano | 1920×1080 clip playback on the desktop |
| `docs/live_demo.md` | — | quick start for the desktop demo |
| `docs/demo_notes.md` | — | this file |

Everything generated at runtime lands in `runs/`, which is gitignored.

---

## Stage 1 — ONNX export (completed, on the x86 workstation)

### Environment

Exported from the workstation conda env `maritime-cv` pinned by
`environment.yml`. No new environment was created for this task and the
Jetson's Python environment was never touched.

| Item | Value |
|---|---|
| Environment | `maritime-cv` |
| Interpreter | Python 3.10.14 |
| `ultralytics` | 8.4.100 |
| `torch` | 2.5.1 (CPU used for export; CUDA irrelevant here) |
| `onnx` | 1.22.0 |
| `onnxslim` | 0.1.95 |
| `onnxruntime-gpu` | 1.23.2 |

Record the `ultralytics` version used for export alongside any comparison to
other artifacts (for example maritime-edge chroma-sweep runs). Do not assume
version parity across experiments without checking.

`onnxslim` was not present initially and was pulled in by ultralytics'
AutoUpdate during the first export (`onnxslim>=0.1.82` is a hard requirement of
`simplify=True`; the requirement check raises before the export runs). The
export script falls back to `simplify=False` if that install is unavailable, and
records which path was taken in the sidecar.

### Source weights

Stock COCO weights already on disk, reused rather than re-downloaded so the file
is known rather than whatever the CDN serves today. The export script looks for a
local `yolo11n.pt` (sibling detector checkout by default) before falling back to
an ultralytics download.

Staged to `runs/yolo11n.pt` before export, because ultralytics writes the ONNX
next to the `.pt` it loaded.

Model as fused for export: 100 layers, 2,616,248 parameters, 6.5 GFLOPs.

### Result

| Item | Value |
|---|---|
| File | `runs/yolo11n.onnx` |
| Size | 10,741,390 bytes (10.74 MB) |
| Opset | 17 |
| Precision | FP32 (FP16 conversion is left to `trtexec`) |
| Input | `images` `[1, 3, 640, 640]` FLOAT |
| Output | `output0` `[1, 84, 8400]` FLOAT |
| Simplify | applied, onnxslim 0.1.95 |
| SHA256 | `1e024ff2e2db5a62de641b89a67fb73a110a01cb2a0070f9cd38488880f02b6a` |

Shapes are fully static, which is what the engine build expects.

The checksum is for verifying the transfer of that recorded file. It is **not** a
reproducibility claim: ultralytics stamps export metadata (including a date)
into the ONNX, so re-exporting the same weights yields a different digest.
Current exports are named `runs/yolo11n_640.onnx` and `runs/yolo11n_1280.onnx`,
each with a matching `yolo11n_<imgsz>_meta.json`. The digest above belongs to
the file recorded then as `runs/yolo11n.onnx`.

### Correctness check before leaving the workstation

The numpy preprocess and postprocess in `argus_trt_infer.py` were validated
against ultralytics' own pipeline on the same ONNX, using onnxruntime in place
of the TensorRT execute call. Five real frames, including one with no
detections:

| Metric | Result |
|---|---|
| Box IoU vs ultralytics | 1.0000 min, 1.0000 mean |
| Confidence delta | 0.00000 max |
| Class agreement | 100% |
| Empty-frame agreement | yes |

That exercises the letterbox geometry (pad 114, centred), the `/255` scaling
with no ImageNet mean/std, confidence as max class score with no objectness
channel, class-wise NMS at IoU 0.45, and the unletterbox back to source
coordinates.

**No timing was taken from this check.** It is a correctness result from an x86
box and nothing more.

Two rooftop captures were staged for stage 3 on the strength of that check. Run
through the ONNX on the workstation, `rooftop_vessel.png` produced one `boat`
detection at confidence 0.543 with a box 204.1 x 74.6 source pixels, and
`rooftop_empty.png` produced none. That makes the pair a usable positive and
negative case for the pixel-cascade segment. These are workstation ONNX
observations recorded to justify the frame choice — the on-device TensorRT
result is a separate measurement and goes in stage 3 once it exists.

---

## Stage 2 — engine build (pending, must run on the Orin Nano)

Not yet run. An engine is compiled for sm_87 and for the TensorRT on the
device. An engine built on the workstation will not load here, so this stage
has no workstation equivalent and no number here can be filled in from
anywhere else.

The build writes one of four engines. The live-demo picture stays 1920×1080;
640 and 1280 are network input sizes.

- `runs/yolo11n_fp16_640.engine`
- `runs/yolo11n_fp16_1280.engine`
- `runs/yolo11n_int8_640.engine`
- `runs/yolo11n_int8_1280.engine`

FP16 is `trtexec --fp16` and needs no Python environment. INT8 runs
`scripts/argus_build_int8.py` from the project venv. It calibrates with
`IInt8EntropyCalibrator2` on stills in `runs/calib/` and writes
`runs/calib_int8_<imgsz>.cache`. Layers without an INT8 kernel fall back to
FP16. That cache is valid only for this TensorRT version on this Nano; delete
it if the ONNX changes. INT8 behaviour depends on those stills and that cache.
Nothing from an INT8 build goes into stage 3 until the build has actually run.

The desktop playback steps are in [`docs/live_demo.md`](live_demo.md).

## Stage 3 — measurements (pending)

**No measurements have been collected yet.** This section is deliberately empty
rather than populated with estimates. Nothing goes here that was not produced by
an actual run on the device. A figure belongs to one engine: its precision and
its imgsz are part of the record, and a 640 FP16 number is not a 1280 INT8
number.

To be recorded once stages 2 and 3 run:

- [ ] Power mode, from `nvpmodel -q`, alongside every figure
- [ ] Engine build time, engine size, engine SHA256, and the build log filename
- [ ] GPU-only latency: p50 / p90 / p99
- [ ] End-to-end latency: p50 / p90 / p99, at a stated synthetic source resolution
- [ ] TensorRT / JetPack / CUDA versions as reported on the device
- [ ] Annotated output frame from a real capture
- [ ] Whether the detector fired on the rooftop vessel frame, and the box height in source pixels

---

## Reproducing

### On the x86 workstation

```bash
conda activate maritime-cv
which python                        # confirm before doing anything else
pip show ultralytics | head -2      # record the version

cd <this-repo>
python scripts/argus_export_onnx.py --imgsz 640
python scripts/argus_export_onnx.py --imgsz 1280
```

### Transfer to the device

`<jetson>` is a stand-in for the SSH destination. No hostname is recorded in
this repo. `<jetson-repo>` is the on-device checkout directory.

```bash
JETSON_DIR=<jetson-repo>

ssh <jetson> "mkdir -p $JETSON_DIR/runs/demo_clips $JETSON_DIR/runs/calib $JETSON_DIR/scripts"
scp runs/yolo11n_640.onnx runs/yolo11n_640_meta.json \
    runs/yolo11n_1280.onnx runs/yolo11n_1280_meta.json \
    runs/rooftop_vessel.png runs/rooftop_empty.png \
    <jetson>:"$JETSON_DIR/runs/"
scp scripts/argus_build_engine.sh scripts/argus_build_int8.py \
    scripts/argus_trt_infer.py scripts/argus_live_demo.py \
    <jetson>:"$JETSON_DIR/scripts/"
```

Stage 1 recorded these digests. The ONNX line is the file written then as
`runs/yolo11n.onnx`. A new export is checked against the sha256 in its own
sidecar, not against that line. The rooftop frame digests are unchanged.

Checksums recorded for the stage 1 artifact:

```
1e024ff2e2db5a62de641b89a67fb73a110a01cb2a0070f9cd38488880f02b6a  runs/yolo11n.onnx
233aec29edf8a4fb3faa8f168f19a49f4f3b13c1e24e0c5b0ae3c7d227c0a02c  runs/yolo11n_meta.json
549b35bd2e2db18306fb2b33e30da817b1765ad72628eaf87fb0647cb88d4df5  runs/rooftop_vessel.png
8917e667d19eb3659f558b4c4dc46dbb2c7a714b30432992b1029c6f4cb46606  runs/rooftop_empty.png
```

### On the Orin Nano

Build an engine. FP16 needs no Python environment — `trtexec` is a compiled
binary, and activating the venv will not fix a failure at that step. INT8 does
need the venv, and stills in `runs/calib/`.

```bash
cd <jetson-repo>
sha256sum runs/yolo11n_640.onnx             # must match that file's sidecar
chmod +x scripts/argus_build_engine.sh
./scripts/argus_build_engine.sh --precision fp16 --imgsz 640
./scripts/argus_build_engine.sh --precision fp16 --imgsz 1280

source .venv/bin/activate
./scripts/argus_build_engine.sh --precision int8 --imgsz 640
./scripts/argus_build_engine.sh --precision int8 --imgsz 1280
```

Each invocation writes `runs/engine_build_<utc>.log`. The default FP16 640
engine path is `runs/yolo11n_fp16_640.engine`. Desktop playback is
[`docs/live_demo.md`](live_demo.md).

Measure. This stage *does* need the venv: it was created with
`--system-site-packages`, which is the only reason `import tensorrt` resolves.

```bash
source .venv/bin/activate
which python                                # confirm before trusting any number
nvpmodel -q                                 # record the power mode

python scripts/argus_trt_infer.py --mode bench
python scripts/argus_trt_infer.py --mode image --source runs/rooftop_vessel.png
python scripts/argus_trt_infer.py --mode image --source runs/rooftop_empty.png
```

### If something fails

- `import tensorrt` fails: the venv is not active, or was rebuilt without
  `--system-site-packages`. Do **not** `pip install tensorrt`; the PyPI wheel is
  x86-oriented and will not work here.
- `torch` prints a compute-capability warning about sm_87 and release 2.13.0:
  expected and benign. `torch.cuda.is_available()` is `True` and kernels run via
  CUDA binary forward compatibility. Do not reinstall torch or set
  `TORCH_CUDA_ARCH_LIST` to silence it.
- The engine fails to deserialize: it was built for a different GPU
  architecture or TensorRT version. Rebuild on the device.
- A stage fails for any other reason: stop and report it. Do not change the
  power mode or run `jetson_clocks` to make a number look better — that
  invalidates the comparison the figures exist to support.

## Invariants that were not touched

`torch 2.13.0+cu132`, CUDA 13.2, TensorRT 10.16.2.10, JetPack 7.2-b187, the
`--system-site-packages` venv, and the 15W power mode are all system-managed and
load-bearing. Nothing in this task modifies any of them, and no `pip install`
runs on the device.

## Out of scope

No training or dataset code, no SAHI or tiling, no tracking, no
containerisation, no web UI. INT8 calibration for the demo engines is in
scope and lives in `scripts/argus_build_int8.py`. Capture and diagnostics
tooling belongs in [maritime-edge](https://github.com/joejyoung/maritime-edge),
not this repository.
