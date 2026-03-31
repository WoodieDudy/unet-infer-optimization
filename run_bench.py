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
    with torch.inference_mode():
        r, mem = bench(model, dummy, warmup, min_run_time, "UNet inference", "fp16 baseline")
        results.append(r)
        peak_mems.append(mem)
        print(r)

    # torch.compile
    compiled = torch.compile(model)
    with torch.inference_mode():
        r, mem = bench(compiled, dummy, warmup, min_run_time, "UNet inference", "torch.compile")
        results.append(r)
        peak_mems.append(mem)
        print(r)

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
