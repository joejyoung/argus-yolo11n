# Live demo quick start

Play short ship clips at 1920×1080 on the Orin Nano desktop, through a TensorRT YOLO11 engine. The window is always 1080p. `640` and `1280` are the network input size of the engine you load, not the size of the picture on the monitor.

Four engines:

| Precision | imgsz | Engine |
|---|---|---|
| FP16 | 640 | `runs/yolo11n_fp16_640.engine` |
| FP16 | 1280 | `runs/yolo11n_fp16_1280.engine` |
| INT8 | 640 | `runs/yolo11n_int8_640.engine` |
| INT8 | 1280 | `runs/yolo11n_int8_1280.engine` |

ONNX stays FP32. FP16 and INT8 are chosen on the device, when the engine is built. An engine built on the workstation will not load on the Nano.

Class names on screen come from the sidecar of the export you built. For the six-class ship weights, name the clips:

```
runs/demo_clips/bulk_cargo_carrier.mp4
runs/demo_clips/container_ship.mp4
runs/demo_clips/fishing_boat.mp4
runs/demo_clips/general_cargo_ship.mp4
runs/demo_clips/ore_carrier.mp4
runs/demo_clips/passenger_ship.mp4
```

A stem that matches a sidecar class (spaces written as `_`) is used as the title. Anything else uses the filename. Clips, calibration stills, ONNX, and engines stay in `runs/` and are not committed.

## 1. Export (x86 workstation)

```bash
conda activate maritime-cv
cd <this-repo>
python scripts/argus_export_onnx.py --imgsz 640
python scripts/argus_export_onnx.py --imgsz 1280
```

Writes `runs/yolo11n_640.onnx`, `runs/yolo11n_640_meta.json`, and the same pair for 1280. Check the sha256 printed at the end of each export. Ultralytics stamps a date into the file, so a later export of the same weights will not match.

## 2. Copy to the Nano

```bash
JETSON_DIR=<jetson-repo>
ssh <jetson> "mkdir -p $JETSON_DIR/runs/demo_clips $JETSON_DIR/runs/calib $JETSON_DIR/scripts"

scp runs/yolo11n_640.onnx runs/yolo11n_640_meta.json \
    runs/yolo11n_1280.onnx runs/yolo11n_1280_meta.json \
    <jetson>:"$JETSON_DIR/runs/"
scp runs/demo_clips/* <jetson>:"$JETSON_DIR/runs/demo_clips/"
scp runs/calib/* <jetson>:"$JETSON_DIR/runs/calib/"
scp scripts/argus_build_engine.sh scripts/argus_build_int8.py \
    scripts/argus_trt_infer.py scripts/argus_live_demo.py \
    <jetson>:"$JETSON_DIR/scripts/"
```

`runs/calib/` is only required for an INT8 build. A handful of representative stills (jpg or png) is enough for a demo cache. On the Nano, confirm each ONNX matches the sha256 in its sidecar.

## 3. Build on the Nano

FP16 needs `trtexec` only. INT8 needs the project venv, because that is how `import tensorrt` resolves. Do not `pip install tensorrt`.

```bash
cd <jetson-repo>
chmod +x scripts/argus_build_engine.sh

./scripts/argus_build_engine.sh --precision fp16 --imgsz 640
./scripts/argus_build_engine.sh --precision fp16 --imgsz 1280

source .venv/bin/activate
./scripts/argus_build_engine.sh --precision int8 --imgsz 640
./scripts/argus_build_engine.sh --precision int8 --imgsz 1280
```

INT8 reads `runs/calib/`, writes `runs/calib_int8_640.cache` or `runs/calib_int8_1280.cache`, and falls back to FP16 for layers that have no INT8 kernel. The cache belongs to this TensorRT version on this Nano. Delete it if you change the ONNX, then build again. Each build appends a provenance log at `runs/engine_build_<utc>.log`. Record `nvpmodel -q` with any number you keep.

## 4. Play on the HDMI desktop

Open a terminal in the logged-in graphical session on the monitor. An SSH shell will not draw there.

```bash
source .venv/bin/activate
python scripts/argus_live_demo.py --precision fp16 --imgsz 640
python scripts/argus_live_demo.py --precision fp16 --imgsz 1280
python scripts/argus_live_demo.py --precision int8 --imgsz 640
python scripts/argus_live_demo.py --precision int8 --imgsz 1280
```

Or point at a file directly:

```bash
python scripts/argus_live_demo.py --engine runs/yolo11n_fp16_1280.engine
```

The window is fullscreen 1920×1080. Frames from a clip that is not already 1080p are letterboxed to that size, then letterboxed again to the engine's `imgsz`.

Keys: `q` quit, `n` next clip, `Space` pause. The playlist loops.

## If something fails

- `import tensorrt` fails: the venv is not active, or it was created without `--system-site-packages`. Do not `pip install tensorrt`.
- The engine fails to deserialize: it was built for another GPU or another TensorRT. Rebuild it on this Nano.
- `engine input ... is not a 640 or 1280 square`, or `--imgsz` disagrees with the file: the engine and the flag are a different size. Pass the `imgsz` that build used.
- `cv2.imshow is not available`: this OpenCV build has no GUI. The demo needs the desktop session on the HDMI output.
- INT8 build stops on an empty `runs/calib/`: add stills. A cache from a different ONNX should be deleted before the rebuild.
- `torch` prints a compute-capability warning about sm_87: expected. Do not reinstall torch to silence it.
- Do not change the power mode or run `jetson_clocks` to make a figure look better.
