"""Генерирует TRT-калибровочный cache на реальных данных, без polygraphy.

Использует чистый TensorRT Python API: парсит ONNX, поднимает builder с
INT8-флагом и `IInt8EntropyCalibrator2`, прогоняет через get_batch
N батчей из test_loader. По завершению `build_serialized_network`
TRT вызывает `write_calibration_cache` → cache на диске.

Сам собранный engine выбрасывается — нам нужен только cache, который
потом скармливается в `trtexec --int8 --calib=<cache>`.
"""

import argparse

import numpy as np
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart

from train import get_loaders


class RealDataCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches, cache_path, input_name="input"):
        super().__init__()
        self.batches = batches
        self.cache_path = cache_path
        self.input_name = input_name
        self.idx = 0
        self.device_input = None
        self.batch_nbytes = batches[0].nbytes
        self._cudart = cudart

    def get_batch_size(self):
        return int(self.batches[0].shape[0])

    def get_batch(self, names):
        if self.idx >= len(self.batches):
            return None
        if self.device_input is None:
            err, self.device_input = cudart.cudaMalloc(self.batch_nbytes)
            assert err == cudart.cudaError_t.cudaSuccess, err
        batch = self.batches[self.idx]
        self.idx += 1
        err = cudart.cudaMemcpy(
            self.device_input, batch.ctypes.data, self.batch_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess, err
        return [int(self.device_input)]

    def read_calibration_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)

    def __del__(self):
        if self.device_input is not None:
            self._cudart.cudaFree(self.device_input)
            self.device_input = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--num-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--input-name", default="input")
    args = p.parse_args()

    _, test_loader = get_loaders(
        input_size=args.input_size, batch_size=args.batch_size, num_workers=2,
    )
    batches = []
    for i, (imgs, _) in enumerate(test_loader):
        if i >= args.num_batches:
            break
        batches.append(np.ascontiguousarray(imgs.numpy(), dtype=np.float32))
    print(f"Loaded {len(batches)} batches × {args.batch_size} = "
          f"{len(batches) * args.batch_size} calibration images")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file умеет подтягивать external data (*.onnx.data рядом)
    if not parser.parse_from_file(args.onnx):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    shape = (args.batch_size, 3, args.input_size, args.input_size)
    profile.set_shape(args.input_name, (1, 3, args.input_size, args.input_size), shape, shape)
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    config.int8_calibrator = RealDataCalibrator(batches, args.cache, args.input_name)

    print("Calibrating (это построит и сразу выбросит engine, нам нужен только cache)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("Build/calibration failed")

    print(f"Cache saved: {args.cache}")


if __name__ == "__main__":
    main()
