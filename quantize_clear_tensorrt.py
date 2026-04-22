import os
from pathlib import Path
import numpy as np
import torch
import tensorrt as trt
import segmentation_models_pytorch as smp
from train import SegmentationModule, get_loaders


# =========================
# TRT LOGGER
# =========================
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

import os
import numpy as np
import tensorrt as trt


# =========================
# CPU CALIBRATOR (NO PYCUDA)
# =========================
class CPUCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file="calib.cache", max_batches=50):
        super().__init__()

        self.dataloader = iter(dataloader)
        self.cache_file = cache_file
        self.max_batches = max_batches
        self.batch_count = 0

        self.current_batch = None

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.batch_count >= self.max_batches:
            return None

        try:
            batch = next(self.dataloader)[0].numpy().astype(np.float32)
        except StopIteration:
            return None

        self.current_batch = batch
        self.batch_count += 1

        # TensorRT expects raw pointer (NO CUDA ALLOC)
        return [batch]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# =========================
# BUILD INT8 ENGINE (REAL)
# =========================
def build_engine_int8(onnx_path, dataloader, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # 🔥 INT8 ON
    config.set_flag(trt.BuilderFlag.INT8)

    # =========================
    # IMPORTANT FIX
    # =========================
    calibrator = CPUCalibrator(dataloader, max_batches=50)
    config.int8_calibrator = calibrator

    print("⏳ Building REAL INT8 engine...")

    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print("✅ Saved:", engine_path)

    return engine

# =========================
# INFERENCE (NO CUDA API, NO PYCUDA)
# =========================
class TRTInfer:
    def __init__(self, engine):
        self.engine = engine
        self.context = engine.create_execution_context()

        self.input_shape = (1, 3, 256, 256)
        self.output_shape = (1, 1, 256, 256)

        self.input = np.zeros(self.input_shape, dtype=np.float32)
        self.output = np.zeros(self.output_shape, dtype=np.float32)

    def __call__(self, x):
        x = x.astype(np.float32)
        np.copyto(self.input, x)

        # ⚠️ TensorRT 8+ requires tensor API OR bindings depending build
        self.context.set_tensor_address("input", self.input.ctypes.data)
        self.context.set_tensor_address("output", self.output.ctypes.data)

        self.context.execute_async_v3(0)

        return self.output.copy()

def main(config):
    exp_dir = Path(config["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = exp_dir / "model.onnx"
    engine_path = exp_dir / "model.trt"

    # ✅ ВОТ ЭТО ОБЯЗАТЕЛЬНО ДОЛЖНО БЫТЬ ДО INT8
    _, test_loader = get_loaders(**config["data"])

    # model
    model_module = SegmentationModule.load_from_checkpoint(
        checkpoint_path=config["model_path"],
        model=smp.Unet(**config["model"]),
    )

    model_module.eval().cpu()

    dummy = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model_module.model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    print("✅ ONNX exported")

    # 🔥 ВОТ ТУТ test_loader уже СУЩЕСТВУЕТ
    engine = build_engine_int8(
        str(onnx_path),
        test_loader,
        str(engine_path)
    )
    print("done")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)