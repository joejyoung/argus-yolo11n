#!/usr/bin/env python3
"""
argus_live_camera.py — run the TensorRT YOLO11 engine on the live USB camera.

The HDMI window is 1920x1080. The network input is whatever the engine was
built for, 640 or 1280, in FP16 or INT8. Capture is the Arducam B0498 on
V4L2 (YUY2 only). nvarguscamerasrc is installed on this Nano and the daemon
is running, but it reports no sensor: this module is USB, not CSI.

    python scripts/argus_live_camera.py --precision fp16 --imgsz 640
    python scripts/argus_live_camera.py --precision int8 --imgsz 1280
    python scripts/argus_live_camera.py --width 3840 --height 2160 --fps 15

Run this from a terminal in the logged-in Nano desktop session. An SSH shell
does not draw on the monitor. Keys: q quit, Space pause.

Exposure is left to v4l2-ctl. Manual mode with exposure_time_absolute=5 is
the cached default and produces a black frame. Set both in one write before
starting, for example:

    v4l2-ctl -d /dev/argus-cam -c auto_exposure=1,exposure_time_absolute=20000
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from argus_live_demo import (
    ALLOWED_IMGSZ,
    detection_line,
    draw_banner,
    engine_imgsz,
    letterbox_display,
    open_window,
    precision_label,
    show,
)
from argus_trt_infer import (
    TRTEngine,
    annotate,
    assert_on_device,
    class_names,
    postprocess,
    preprocess,
    require_cv2,
)

ARGUS_CAM = "/dev/argus-cam"
VIDEO0 = "/dev/video0"


def default_device():
    if os.path.exists(ARGUS_CAM):
        return ARGUS_CAM
    return VIDEO0


def reject_metadata_node(device):
    """video1 on this UVC camera carries payload headers, not pictures."""
    node = os.path.basename(os.path.realpath(device))
    if node == "video1":
        sys.exit(
            f"{device} is the UVC metadata node, not the picture.\n"
            "Use /dev/argus-cam or /dev/video0. "
            "v4l2-ctl --list-devices shows which node is which."
        )


def capture_pipeline(device, width, height, fps):
    """YUY2 from v4l2src, newest frame only, BGR for OpenCV.

    The queue drops the older buffer when the next frame arrives, so v4l2src
    keeps returning USB buffers while inference and videoconvert are busy.
    Byte and time limits are zeroed so the queue's defaults cannot hold a backlog.
    """
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"video/x-raw,format=YUY2,width={width},height={height},"
        f"framerate={fps}/1 ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 "
        "leaky=downstream ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def open_camera(device, width, height, fps):
    cv = require_cv2()
    if not hasattr(cv, "CAP_GSTREAMER"):
        sys.exit(
            "this OpenCV build has no GStreamer backend.\n"
            "The camera demo opens the Arducam through v4l2src."
        )
    if not os.path.exists(device):
        sys.exit(
            f"camera node not found: {device}\n"
            "Check the node with: v4l2-ctl --list-devices"
        )
    reject_metadata_node(device)
    pipeline = capture_pipeline(device, width, height, fps)
    cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit(
            f"could not open {device} at {width}x{height}@{fps} YUY2.\n"
            "The Arducam publishes YUYV only: 3840x2160 at 15/10/5, "
            "1920x1080 at 60/30/15, 1280x720 at 90/60/30/15.\n"
            "This OpenCV needs a GStreamer backend. "
            "Check the node with: v4l2-ctl --list-devices"
        )
    return cap


def play_camera(cap, device, engine, names, size, title, tag, conf, iou):
    """Show the camera until quit. A failed read ends the demo."""
    paused = False
    vis = None
    last = time.perf_counter()
    fps = 0.0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                sys.exit(f"camera stopped producing frames: {device}")
            shown = letterbox_display(frame)
            x, r, dx, dy = preprocess(shown, size, engine.input_dtype)
            dets = postprocess(
                engine.infer(x), r, dx, dy, shown.shape, conf, iou)
            now = time.perf_counter()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            vis = annotate(shown, dets, names)
            detail = f"{tag}{size}   {fps:.0f} fps   {detection_line(dets, names)}"
            vis = draw_banner(vis, title, detail)
        show(vis)
        raw = require_cv2().waitKey(0 if paused else 1)
        if raw < 0:
            continue
        key = raw & 0xFF
        if key in (ord("q"), 27):
            return
        if key == ord(" "):
            paused = not paused


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    p.add_argument("--imgsz", type=int, choices=ALLOWED_IMGSZ, default=None,
                   help="engine spatial size, 640 or 1280")
    p.add_argument("--engine", default="",
                   help="engine path (default runs/yolo11n_<precision>_<imgsz>.engine)")
    p.add_argument("--meta", default="",
                   help="sidecar (default runs/yolo11n_<imgsz>_meta.json)")
    p.add_argument("--device", default="",
                   help="V4L2 node (default /dev/argus-cam, else /dev/video0)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    args = p.parse_args()

    if args.width < 1 or args.height < 1 or args.fps < 1:
        sys.exit("--width, --height, and --fps must be positive")

    assert_on_device()

    device = args.device or default_device()
    imgsz = args.imgsz if args.imgsz is not None else 640
    engine_path = args.engine or f"runs/yolo11n_{args.precision}_{imgsz}.engine"
    if not os.path.isfile(engine_path):
        sys.exit(
            f"engine not found: {engine_path}\n"
            "Build it on this Nano with argus_build_engine.sh."
        )

    engine = TRTEngine(engine_path)
    size = engine_imgsz(engine, args.imgsz)
    meta_path = args.meta or f"runs/yolo11n_{size}_meta.json"
    names = class_names(meta_path)
    tag = precision_label(engine_path)
    tag = f"{tag}  " if tag else ""
    title = f"live  {device}"

    print(f"engine   {engine_path}")
    print(f"input    {size}x{size}")
    print(f"meta     {meta_path}")
    print(f"camera   {device}  {args.width}x{args.height}@{args.fps} YUY2")
    print("window   1920x1080 fullscreen")
    print("keys     q quit    Space pause")

    cap = open_camera(device, args.width, args.height, args.fps)
    try:
        open_window()
        play_camera(
            cap, device, engine, names, size, title, tag, args.conf, args.iou)
    finally:
        cap.release()
        require_cv2().destroyAllWindows()


if __name__ == "__main__":
    main()
