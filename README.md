# argus-yolo11n

Stock YOLOv11n deployment path for the Jetson Orin Nano.

```
yolo11n.pt  →  ONNX (opset 17, static 640)  →  trtexec  →  TensorRT FP16 (sm_87)  →  Python inference
```

This repository exercises the **edge deployment path** with stock COCO weights. It is not a maritime detector, and detections on maritime scenes are not a performance claim about the capstone system.

Full run notes, checksums, and boundaries live in [`docs/demo_notes.md`](docs/demo_notes.md).

---

## Status

| Stage | Host | State |
|---|---|---|
| 1. Export ONNX + metadata | x86 workstation | Done |
| 2. Build TensorRT engine | Orin Nano | Pending |
| 3. On-device latency / image runs | Orin Nano | Pending |

Latency figures are only meaningful next to the device power mode that produced them. Capture-chain measurements (chroma sweep) are a different experiment and must not be compared side by side with on-device TensorRT latency.

---

## Layout

```
scripts/          export, engine build, inference, and Argus diagnostics
docs/demo_notes.md
runs/             gitignored runtime artifacts (ONNX, engine, logs, frames)
```

### Deployment path

| Script | Host | Role |
|---|---|---|
| `scripts/argus_export_onnx.py` | x86 | `.pt` → ONNX + `yolo11n_meta.json` |
| `scripts/argus_build_engine.sh` | Orin Nano | ONNX → FP16 engine + build provenance log |
| `scripts/argus_trt_infer.py` | Orin Nano | `--mode image` / `--mode bench` |

### Capture and diagnostics

Also included: session capture, camera/optics checks, frame stats, chroma-sweep ablation, AIS correlation helpers, JetPack/integrity audits (`argus-session.sh`, `argus_camcheck.sh`, `argus_opticheck.sh`, `chroma_sweep.py`, and related scripts).

---

## Quick start

### 1. Export (x86)

Same environment that runs `chroma_sweep.py`. Do not install ultralytics into the Jetson venv.

```bash
cd <this-repo>
python scripts/argus_export_onnx.py
# or: python scripts/argus_export_onnx.py --weights /path/to/yolo11n.pt
```

Writes `runs/yolo11n.onnx` and `runs/yolo11n_meta.json`.

### 2. Transfer to the device

```bash
JETSON_DIR=<jetson-repo>
ssh <jetson> "mkdir -p $JETSON_DIR/runs $JETSON_DIR/scripts"
scp runs/yolo11n.onnx runs/yolo11n_meta.json <jetson>:"$JETSON_DIR/runs/"
scp scripts/argus_build_engine.sh scripts/argus_trt_infer.py <jetson>:"$JETSON_DIR/scripts/"
```

### 3. Build engine (Orin Nano)

Needs `trtexec` only — no Python venv for this step. An engine built on the workstation will not load on the device.

```bash
cd <jetson-repo>
./scripts/argus_build_engine.sh
```

### 4. Infer / bench (Orin Nano)

Requires the project venv created with `--system-site-packages` so `import tensorrt` resolves. Do not `pip install tensorrt`.

```bash
source .venv/bin/activate
nvpmodel -q    # record alongside every figure

python scripts/argus_trt_infer.py --mode bench
python scripts/argus_trt_infer.py --mode image --source runs/rooftop_vessel.png
```

---

## Invariants

On-device measurements assume JetPack-managed TensorRT/CUDA/torch and a recorded power mode (typically 15 W). Nothing in this path should reinstall those packages or silence compute-capability warnings by changing the torch build.

See [`docs/demo_notes.md`](docs/demo_notes.md) for environment versions, transfer checksums, and failure modes.
