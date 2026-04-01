# UNet Inference Optimization

Бенчмарк оптимизаций инференса UNet

Подробный отчёт: [report.md](report.md)

## Команда

* Крамаренко Георгий
* Яскевич Михаил
* Липилин Матвей

## План работы

- 30.03-05.04 - Выбрать датасет и прогнать валидацию на нём
- 06.04-12.04 - Добавить сравнение с альтернативным компилятором. И квантизацию.
- 13.04-19.04 - Добавить спарсификацию. Сделать презу.

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
pip install -r requirements.txt
python run_bench.py
```

Доступные параметры:

```bash
python run_bench.py --batch-size 16 --input-size 512 --min-run-time 5 --warmup 30
```

Результаты автоматически сохраняются в `results.json`.


## Результаты

| Pipeline | Latency (ms) | Throughput (img/s) | Peak GPU (MB) | Speedup |
|---|---|---|---|---|
| fp16 baseline | 35.47 | 902.3 | — | 1.00x |
| torch.compile | 20.81 | 1537.9 | — | 1.70x |