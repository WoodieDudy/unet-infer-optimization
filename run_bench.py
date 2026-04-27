import argparse
import json
from datetime import datetime, timezone

import torch
from torch.utils.benchmark import Timer
import segmentation_models_pytorch as smp

DEVICE = "cuda"
DTYPE = torch.float16


def parse_args():
    p = argparse.ArgumentParser(description="UNet inference benchmark")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--min-run-time", type=float, default=3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--trt-engine",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="TRT engine to bench, format NAME=PATH. Можно указывать несколько раз.",
    )
    return p.parse_args()


def print_env_info(input_shape):
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  CUDA (runtime) : {torch.version.cuda}")
    print(f"  cuDNN          : {torch.backends.cudnn.version()}")
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Input shape    : {tuple(input_shape)}")
    print(f"  Dtype          : {DTYPE}")
    print("=" * 60)
    print()


def get_env_dict(input_shape):
    return {
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "input_shape": list(input_shape),
        "dtype": str(DTYPE),
    }


class TRTComputeRunner:
    """Compute-only TRT обёртка: input/output живут в GPU-памяти, передаём указатели через set_tensor_address."""
    def __init__(self, engine_path, input_shape, input_name="input", output_name="output", device="cuda"):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._cudart = cudart
        self._input_name = input_name
        self._output_name = output_name

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine from {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TRT execution context")
        self._context.set_input_shape(input_name, tuple(input_shape))

        out_shape = tuple(self._context.get_tensor_shape(output_name))
        self._output = torch.empty(out_shape, dtype=torch.float32, device=device)
        self._context.set_tensor_address(output_name, self._output.data_ptr())

        err, self._stream = cudart.cudaStreamCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._context.set_tensor_address(self._input_name, x.data_ptr())
        ok = self._context.execute_async_v3(self._stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 failed")
        (err,) = self._cudart.cudaStreamSynchronize(self._stream)
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")
        return self._output

    def close(self):
        if self._stream is not None:
            self._cudart.cudaStreamDestroy(self._stream)
            self._stream = None


def bench(fn, dummy, warmup: int, min_run_time: float, label: str, description: str):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(dummy)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    timer = Timer(
        stmt="fn(x)",
        globals={"fn": fn, "x": dummy},
        label=label,
        description=description,
    )
    result = timer.blocked_autorange(min_run_time=min_run_time)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return result, peak_mb


def main():
    args = parse_args()
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    min_run_time = args.min_run_time
    warmup = args.warmup

    print_env_info(input_shape)

    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model = model.to(DEVICE, dtype=DTYPE).eval()

    torch.manual_seed(42)
    dummy = torch.randn(input_shape, device=DEVICE, dtype=DTYPE)

    results = []
    peak_mems = []

    # fp16 baseline
    r, mem = bench(model, dummy, warmup, min_run_time, "UNet inference", "fp16 baseline")
    results.append(r)
    peak_mems.append(mem)
    print(r)

    # torch.compile
    compiled = torch.compile(model)
    r, mem = bench(compiled, dummy, warmup, min_run_time, "UNet inference", "torch.compile")
    results.append(r)
    peak_mems.append(mem)
    print(r)

    # TRT engines (compute-only через сырой TRT API + zero-copy data_ptr())
    trt_runners = []
    try:
        if args.trt_engine:
            dummy_fp32 = dummy.float().contiguous()  # engine ждёт fp32 input
        for spec in args.trt_engine:
            if "=" not in spec:
                raise SystemExit(f"--trt-engine ожидает NAME=PATH, получено: {spec!r}")
            name, path = spec.split("=", 1)
            runner = TRTComputeRunner(path, input_shape)
            trt_runners.append(runner)
            r, mem = bench(runner, dummy_fp32, warmup, min_run_time, "UNet inference", name)
            results.append(r)
            peak_mems.append(mem)
            print(r)
    finally:
        for runner in trt_runners:
            runner.close()

    compare = torch.utils.benchmark.Compare(results)
    print()
    compare.print()

    batch_size = input_shape[0]
    baseline_median = results[0].median

    print("| Pipeline | Latency (ms) | Throughput (img/s) | Peak GPU (MB) | Speedup |")
    print("|---|---|---|---|---|")
    json_rows = []
    for r, mem in zip(results, peak_mems):
        latency_ms = r.median * 1000
        throughput = batch_size / r.median
        speedup = baseline_median / r.median
        print(f"| {r.description} | {latency_ms:.2f} | {throughput:.1f} | {mem:.0f} | {speedup:.2f}x |")
        json_rows.append({
            "pipeline": r.description,
            "latency_ms": latency_ms,
            "throughput_img_s": throughput,
            "peak_gpu_mb": mem,
            "speedup": speedup,
        })

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": get_env_dict(input_shape),
        "results": json_rows,
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results.json")


if __name__ == "__main__":
    main()
