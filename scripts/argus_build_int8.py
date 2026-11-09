#!/usr/bin/env python3
"""
argus_build_int8.py — ONNX -> TensorRT INT8 engine, on-device.

Called by argus_build_engine.sh when --precision int8 is set. Do not point
this at a workstation: the engine is compiled for the GPU that builds it.

Calibration uses IInt8EntropyCalibrator2. Each image is letterboxed and
scaled with the same preprocess as inference, at the ONNX spatial size.
Layers without an INT8 kernel fall back to FP16. The calibration cache is
valid only for this TensorRT build on this device; delete it if the ONNX
changes.

Needs the project venv (system TensorRT and torch). Do not pip install
tensorrt.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from argus_trt_infer import assert_on_device, preprocess, require_cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(directory):
    if not os.path.isdir(directory):
        sys.exit(f"calibration directory not found: {directory}\n"
                 "Put representative stills there (jpg, png, or bmp). "
                 "An empty directory cannot build an INT8 engine.")
    paths = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            paths.append(path)
    if not paths:
        sys.exit(f"no calibration images in {directory}\n"
                 "Put representative stills there (jpg, png, or bmp).")
    return paths


def make_calibrator(trt, torch, images, imgsz, cache_path):
    """Entropy calibrator over letterboxed FP32 batches. Batch size is 1."""

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.images = images
            self.index = 0
            self.cache_path = cache_path
            self.device = torch.empty(
                (1, 3, imgsz, imgsz), dtype=torch.float32, device="cuda")

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            cv = require_cv2()
            while self.index < len(self.images):
                path = self.images[self.index]
                self.index += 1
                frame = cv.imread(path, cv.IMREAD_COLOR)
                if frame is None:
                    print(f"warning: could not decode {path}", file=sys.stderr)
                    continue
                batch, _, _, _ = preprocess(frame, imgsz, np.float32)
                host = torch.from_numpy(np.ascontiguousarray(batch))
                self.device.copy_(host)
                return [int(self.device.data_ptr())]
            return None

        def read_calibration_cache(self):
            if os.path.isfile(self.cache_path) and os.path.getsize(self.cache_path) > 0:
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            parent = os.path.dirname(self.cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                f.write(cache)
            print(f"calibration cache  {self.cache_path}")

    return Calibrator()


def build(args):
    try:
        import tensorrt as trt
    except ImportError as e:
        sys.exit(f"import tensorrt failed ({e}).\n"
                 "Activate the project venv before an INT8 build. "
                 "Do not pip install tensorrt.")
    try:
        import torch
    except ImportError as e:
        sys.exit(f"import torch failed ({e}). Activate the project venv.")

    if not torch.cuda.is_available():
        sys.exit("torch reports no CUDA device; cannot calibrate or build.")

    print(f"TensorRT version: {trt.__version__}")
    print(f"torch             {torch.__version__}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if not builder.platform_has_fast_int8:
        print("warning: this platform reports no fast INT8 path", file=sys.stderr)

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        blob = f.read()
    if not parser.parse(blob):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        sys.exit("failed to parse ONNX")

    if network.num_inputs != 1:
        sys.exit(f"expected 1 ONNX input, got {network.num_inputs}")
    shape = tuple(int(d) for d in network.get_input(0).shape)
    expect = (1, 3, args.imgsz, args.imgsz)
    if shape != expect:
        sys.exit(f"ONNX input {list(shape)} does not match --imgsz {args.imgsz}")

    images = list_images(args.calib_dir)
    print(f"calibration images  {len(images)} from {args.calib_dir}")
    if os.path.isfile(args.cache) and os.path.getsize(args.cache) > 0:
        print(f"reusing cache       {args.cache}")
        print("                    delete it if this ONNX is not the one that wrote it")

    # Held until build_serialized_network returns. TensorRT calls into this
    # object during the build; dropping the reference early crashes it.
    calibrator = make_calibrator(trt, torch, images, args.imgsz, args.cache)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 20))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    print(f"building            {args.engine}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit("INT8 build produced no engine")

    parent = os.path.dirname(args.engine)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.engine, "wb") as f:
        f.write(bytes(serialized))
    print(f"engine              {args.engine}  ({os.path.getsize(args.engine)} bytes)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--imgsz", type=int, choices=(640, 1280), required=True)
    p.add_argument("--workspace", type=int, default=2048,
                   help="workspace pool in MB")
    p.add_argument("--calib-dir", default="runs/calib")
    p.add_argument("--cache", required=True, help="INT8 calibration cache path")
    args = p.parse_args()

    if args.workspace < 1:
        sys.exit(f"--workspace wants a positive number of MB, got {args.workspace}")
    if not os.path.isfile(args.onnx):
        sys.exit(f"ONNX not found: {args.onnx}")

    assert_on_device()
    build(args)


if __name__ == "__main__":
    main()
