import torch
from torch.utils.benchmark import Timer
import segmentation_models_pytorch as smp

DEVICE = "cuda"
DTYPE = torch.float16
INPUT_SHAPE = (32, 3, 256, 256)
MIN_RUN_TIME = 3


model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
model = model.to(DEVICE, dtype=DTYPE).eval()

dummy = torch.randn(INPUT_SHAPE, device=DEVICE, dtype=DTYPE)


def bench(fn, label: str, description: str):
    # warmup
    with torch.inference_mode():
        for _ in range(20):
            fn(dummy)
    torch.cuda.synchronize()

    timer = Timer(
        stmt="fn(x)",
        globals={"fn": fn, "x": dummy},
        label=label,
        description=description,
    )
    result = timer.blocked_autorange(min_run_time=MIN_RUN_TIME)
    return result


results = []

# fp16 baseline
with torch.inference_mode():
    r = bench(model, "UNet inference", "fp16 baseline")
    results.append(r)
    print(r)

# torch.compile
compiled = torch.compile(model)
with torch.inference_mode():
    r = bench(compiled, "UNet inference", "torch.compile")
    results.append(r)
    print(r)

compare = torch.utils.benchmark.Compare(results)
print()
compare.print()

batch_size = INPUT_SHAPE[0]
baseline_median = results[0].median

print(f"| Pipeline | Latency (ms) | Throughput (img/s) | Speedup |")
print(f"|---|---|---|---|")
for r in results:
    latency_ms = r.median * 1000
    throughput = batch_size / r.median
    speedup = baseline_median / r.median
    print(f"| {r.description} | {latency_ms:.2f} | {throughput:.1f} | {speedup:.2f}x |")
