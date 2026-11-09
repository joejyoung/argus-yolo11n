#!/usr/bin/env python3
"""
argus_live_demo.py — play 1080p ship clips through a TensorRT YOLO11 engine.

The HDMI window is 1920x1080. The network input is whatever the engine was
built for, 640 or 1280, in FP16 or INT8. Clips are files, not the camera.

    python scripts/argus_live_demo.py --precision fp16 --imgsz 640
    python scripts/argus_live_demo.py --precision int8 --imgsz 1280
    python scripts/argus_live_demo.py --engine runs/yolo11n_fp16_1280.engine

Run this from a terminal in the logged-in Nano desktop session. An SSH shell
does not draw on the monitor. Keys: q quit, n next clip, Space pause.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from argus_trt_infer import (
    TRTEngine,
    annotate,
    assert_on_device,
    class_names,
    postprocess,
    preprocess,
    require_cv2,
)

DISPLAY_W, DISPLAY_H = 1920, 1080
ALLOWED_IMGSZ = (640, 1280)
WINDOW = "argus"
CLIP_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
PAD = 114


def letterbox_display(img, width=DISPLAY_W, height=DISPLAY_H, pad=PAD):
    """Fit a frame into 1920x1080. Inference letterboxes again to the engine."""
    cv = require_cv2()
    h, w = img.shape[:2]
    r = min(width / w, height / h)
    nw = min(width, int(round(w * r)))
    nh = min(height, int(round(h * r)))
    out = np.full((height, width, 3), pad, dtype=np.uint8)
    if (nw, nh) != (w, h):
        img = cv.resize(img, (nw, nh), interpolation=cv.INTER_LINEAR)
    dx, dy = (width - nw) // 2, (height - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = img
    return out


def list_clips(directory):
    if not os.path.isdir(directory):
        sys.exit(f"clip directory not found: {directory}")
    clips = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in CLIP_EXTS:
            clips.append(path)
    if not clips:
        sys.exit(
            f"no clips in {directory}\n"
            "Expected short 1920x1080 files, one per class, for example:\n"
            "  bulk_cargo_carrier.mp4\n"
            "  container_ship.mp4\n"
            "  fishing_boat.mp4\n"
            "  general_cargo_ship.mp4\n"
            "  ore_carrier.mp4\n"
            "  passenger_ship.mp4"
        )
    return clips


def clip_title(path, names):
    stem = os.path.splitext(os.path.basename(path))[0]
    key = stem.replace("_", " ").lower()
    for label in names.values():
        if str(label).lower() == key:
            return str(label)
    return stem.replace("_", " ")


def precision_label(engine_path):
    base = os.path.basename(engine_path).lower()
    for name in ("fp16", "int8"):
        if name in base:
            return name
    return ""


def engine_imgsz(engine, requested):
    shape = tuple(int(d) for d in engine.input_shape)
    if len(shape) != 4 or shape[2] != shape[3] or shape[2] not in ALLOWED_IMGSZ:
        sys.exit(
            f"engine input {list(shape)} is not a 640 or 1280 square.\n"
            "Rebuild with argus_build_engine.sh --imgsz 640 or --imgsz 1280."
        )
    size = shape[2]
    if requested is not None and requested != size:
        sys.exit(
            f"--imgsz {requested} does not match the engine input {size}.\n"
            "Pass the imgsz the engine was built with, or omit --imgsz when "
            "selecting an engine by path."
        )
    return size


def open_window():
    cv = require_cv2()
    try:
        cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(WINDOW, DISPLAY_W, DISPLAY_H)
        cv.setWindowProperty(WINDOW, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    except cv.error as e:
        gui_missing(e)


def show(frame):
    cv = require_cv2()
    try:
        cv.imshow(WINDOW, frame)
    except cv.error as e:
        gui_missing(e)


def gui_missing(exc):
    if "not implemented" in str(exc).lower():
        sys.exit(
            "cv2.imshow is not available in this OpenCV build.\n"
            "This demo needs the desktop GUI on the HDMI display. "
            "The JetPack OpenCV here was built without a windowing backend."
        )
    raise exc


def draw_banner(img, title, detail):
    cv = require_cv2()
    scale = max(1.0, img.shape[0] / 900)
    thick = max(2, round(img.shape[0] / 500))
    y = int(72 * scale)
    for line in (title, detail):
        (tw, th), base = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv.rectangle(img, (16, y - th - 8), (32 + tw, y + base + 8), (0, 0, 0), -1)
        cv.putText(img, line, (24, y), cv.FONT_HERSHEY_SIMPLEX, scale,
                   (255, 255, 255), thick, cv.LINE_AA)
        y += th + base + int(20 * scale)
    return img


def detection_line(dets, names):
    if len(dets) == 0:
        return "no detection"
    x1, y1, x2, y2, conf, cls = dets[0]
    label = names.get(int(cls), f"class_{int(cls)}")
    extra = f"  +{len(dets) - 1}" if len(dets) > 1 else ""
    return f"{len(dets)} det  {label} {conf:.2f}{extra}"


def play_clip(path, engine, names, size, title, tag, conf, iou):
    """Play one clip. Returns 'quit' or 'next'."""
    cv = require_cv2()
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open clip: {path}")

    paused = False
    vis = None
    last = time.perf_counter()
    fps = 0.0
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if vis is None:
                        sys.exit(f"could not decode {path}")
                    return "next"
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
            raw = cv.waitKey(0 if paused else 1)
            if raw < 0:
                continue
            key = raw & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            if key == ord("n"):
                return "next"
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()


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
    p.add_argument("--clips", default="runs/demo_clips")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    args = p.parse_args()

    assert_on_device()

    imgsz = args.imgsz if args.imgsz is not None else 640
    engine_path = args.engine or f"runs/yolo11n_{args.precision}_{imgsz}.engine"
    if not os.path.isfile(engine_path):
        sys.exit(
            f"engine not found: {engine_path}\n"
            "Build it on this Nano with argus_build_engine.sh."
        )

    clips = list_clips(args.clips)
    engine = TRTEngine(engine_path)
    size = engine_imgsz(engine, args.imgsz)
    meta_path = args.meta or f"runs/yolo11n_{size}_meta.json"
    names = class_names(meta_path)
    tag = precision_label(engine_path)
    tag = f"{tag}  " if tag else ""

    print(f"engine   {engine_path}")
    print(f"input    {size}x{size}")
    print(f"meta     {meta_path}")
    print(f"clips    {len(clips)} from {args.clips}")
    print("window   1920x1080 fullscreen")
    print("keys     q quit    n next clip    Space pause")

    open_window()
    index = 0
    try:
        while True:
            path = clips[index]
            title = clip_title(path, names)
            print(f"play     {path}  ({title})")
            action = play_clip(
                path, engine, names, size, title, tag, args.conf, args.iou)
            if action == "quit":
                break
            index = (index + 1) % len(clips)
    finally:
        require_cv2().destroyAllWindows()


if __name__ == "__main__":
    main()
