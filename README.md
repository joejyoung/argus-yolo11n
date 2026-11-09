# argus-yolo11n

Stock YOLOv11n deployment path for the Jetson Orin Nano.

```
yolo11n.pt  →  ONNX (opset 17, static 640 or 1280)  →  TensorRT FP16 or INT8 (sm_87)  →  Python inference
```

This repository exercises the **edge deployment path** with stock COCO weights. It is not a maritime detector, and detections on maritime scenes are not a performance claim about the capstone system.

Full run notes, checksums, and boundaries live in [`docs/demo_notes.md`](docs/demo_notes.md). The HDMI clip player is [`docs/live_demo.md`](docs/live_demo.md).

---

## Status

| Stage | Host | State |
|---|---|---|
| 1. Export ONNX + metadata | x86 workstation | Done |
| 2. Build TensorRT engine | Orin Nano | Pending |
| 3. On-device latency / image runs | Orin Nano | Pending |

Latency figures are only meaningful next to the device power mode that produced them. Capture-chain measurements (chroma sweep) live in [maritime-edge](https://github.com/joejyoung/maritime-edge) and must not be compared side by side with on-device TensorRT latency.

---

## Layout

```
scripts/          export, engine build, inference, and the desktop demo
docs/demo_notes.md
docs/live_demo.md
runs/             gitignored runtime artifacts (ONNX, engine, logs, frames, clips)
```

### Deployment path

| Script | Host | Role |
|---|---|---|
| `scripts/argus_export_onnx.py` | x86 | `.pt` → ONNX + `yolo11n_<imgsz>_meta.json` |
| `scripts/argus_build_engine.sh` | Orin Nano | ONNX → FP16 or INT8 engine + build provenance log |
| `scripts/argus_build_int8.py` | Orin Nano | INT8 calibrator, called by the build script |
| `scripts/argus_trt_infer.py` | Orin Nano | `--mode image` / `--mode bench` |
| `scripts/argus_live_demo.py` | Orin Nano | 1920×1080 clip playback on the desktop |

### Capture and diagnostics

Session capture, camera/optics checks, frame stats, chroma-sweep ablation, AIS correlation helpers, and JetPack/integrity audits live in the parallel [maritime-edge](https://github.com/joejyoung/maritime-edge) repository (`argus-session.sh`, `argus_camcheck.sh`, `argus_opticheck.sh`, `chroma_sweep.py`, and related scripts).

---

## Environment (x86)

Recreate the workstation conda env used for ONNX export:

```bash
conda env create -f environment.yml
conda activate maritime-cv
```

`requirements.txt` pins the same pip packages if you already have a compatible PyTorch CUDA 12.1 install. Neither file is for the Orin Nano — on-device inference uses JetPack system TensorRT/torch.

## Quick start

### 1. Export (x86)

```bash
conda activate maritime-cv
cd <this-repo>
python scripts/argus_export_onnx.py --imgsz 640
python scripts/argus_export_onnx.py --imgsz 1280
# or: python scripts/argus_export_onnx.py --imgsz 640 --weights /path/to/yolo11n.pt
```

Writes `runs/yolo11n_640.onnx` and `runs/yolo11n_640_meta.json`, and the same pair for 1280.

### 2. Transfer to the device

```bash
JETSON_DIR=<jetson-repo>
ssh <jetson> "mkdir -p $JETSON_DIR/runs $JETSON_DIR/scripts"
scp runs/yolo11n_640.onnx runs/yolo11n_640_meta.json \
    runs/yolo11n_1280.onnx runs/yolo11n_1280_meta.json \
    <jetson>:"$JETSON_DIR/runs/"
scp scripts/argus_build_engine.sh scripts/argus_build_int8.py \
    scripts/argus_trt_infer.py scripts/argus_live_demo.py \
    <jetson>:"$JETSON_DIR/scripts/"
```

### 3. Build engine (Orin Nano)

FP16 needs `trtexec` only. INT8 needs the project venv and stills in `runs/calib/`. An engine built on the workstation will not load on the device. The four build commands and the desktop player are in [`docs/live_demo.md`](docs/live_demo.md).

```bash
cd <jetson-repo>
./scripts/argus_build_engine.sh --precision fp16 --imgsz 640
```

### 4. Infer / bench (Orin Nano)

Requires the project venv created with `--system-site-packages` so `import tensorrt` resolves. Do not `pip install tensorrt`.

```bash
source .venv/bin/activate
nvpmodel -q    # record alongside every figure

python scripts/argus_trt_infer.py --mode bench
python scripts/argus_trt_infer.py --mode image --source runs/rooftop_vessel.png
python scripts/argus_live_demo.py --precision fp16 --imgsz 640
```

---

## Invariants

On-device measurements assume JetPack-managed TensorRT/CUDA/torch and a recorded power mode (typically 15 W). Nothing in this path should reinstall those packages or silence compute-capability warnings by changing the torch build.

See [`docs/demo_notes.md`](docs/demo_notes.md) for environment versions, transfer checksums, and failure modes.
