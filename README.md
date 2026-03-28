# UNet Inference Optimization

Бенчмарк оптимизаций инференса UNet

## Окружение

- **GPU:** NVIDIA A100-PCIE-40GB (MIG 2g.10gb)
- **CUDA:** 12.8
- **PyTorch:** 2.11.0

## Docker

```bash
docker build -t unet-bench .

docker run --rm --gpus all unet-bench
```

## Запуск без Docker

```bash
pip install torch torchvision segmentation-models-pytorch
python run_bench.py
```


## Результаты

| Pipeline | Latency (ms) | Throughput (img/s) | Speedup |
|---|---|---|---|
| fp16 baseline | 35.47 | 902.3 | 1.00x |
| torch.compile | 20.81 | 1537.9 | 1.70x |