# UNet Inference Optimization

Бенчмарк оптимизаций инференса UNet

## Команда

* Крамаренко Георгий
* Яскевич Михаил
* Липилин Матвей

## Окружение

| Компонент | Версия |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB (полный, без MIG) |
| Драйвер / CUDA | 580.126 / 13.2 |
| TensorRT | 10.16.1.11 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12 |


## Установка

```bash
# нужны для visualize_engines.py через trex)
apt-get install -y graphviz

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Датасет

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 пород, ~7400 изображений с масками сегментации (3 класса: питомец / фон / граница).
Качается автоматически при первом запуске `train.py` через `torchvision.datasets.OxfordIIITPet`


## 1. Обучение

```bash
python train.py --config_path train_config_local.yaml
```

Параметры в [train_config_local.yaml](train_config_local.yaml): 10 эпох, batch 32, input 256, lr 1e-4, precision `16-mixed`.
Чекпоинт: `experiments/train_local/lightning_logs/version_0/checkpoints/trained_baseline.ckpt`
Финальный `val_iou ≈ 0.873`.


## 1.2 Демо-инференс на реальных картинках

Чтобы убедиться, что обученный чекпоинт действительно сегментирует, визуализируем предсказания.

```bash
python inference_demo.py \
    --ckpt experiments/train_local/lightning_logs/version_0/checkpoints/trained_baseline.ckpt \
    --out experiments/inference_demo.png \
    --num-images 4
```

![inference demo](experiments/inference_demo.png)

Видно, что предсказания почти точно совпадают с ground truth (`val_iou ≈ 0.873`).


## 2. Экспорт в ONNX

```bash
python export_onnx.py \
    --ckpt experiments/train_local/lightning_logs/version_0/checkpoints/trained_baseline.ckpt \
    --onnx experiments/trt_fp16/unet.onnx
```

## 3. Сторонний движок. Сборка TRT engine (fp16)

```bash
trtexec --onnx=experiments/trt_fp16/unet.onnx \
    --saveEngine=experiments/trt_fp16/unet_fp16.engine \
    --fp16 \
    --minShapes=input:1x3x256x256 \
    --optShapes=input:32x3x256x256 \
    --maxShapes=input:32x3x256x256 \
    --profilingVerbosity=detailed \
    --skipInference
```

- `--profilingVerbosity=detailed` нужен для последующей визуализации через `trex`.

## 4. Сборка TRT engine (int8) с калибровкой на реальных данных

`trtexec --int8` без указания калибровочного cache берёт **случайный шум** для статистик активаций — так делать нельзя. Сначала генерируем cache на реальных данных через `calibrate.py`, потом скармливаем cache в `trtexec`.

```bash
# 1. Калибровка: 16 батчей × 32 картинки из test_loader → calib.cache
python3 calibrate.py \
    --onnx experiments/trt_fp16/unet.onnx \
    --cache experiments/trt_fp16/calib_dense.cache

# 2. Build int8 engine с этим cache
trtexec --onnx=experiments/trt_fp16/unet.onnx \
    --saveEngine=experiments/trt_fp16/unet_int8.engine \
    --fp16 --int8 --calib=experiments/trt_fp16/calib_dense.cache \
    --minShapes=input:1x3x256x256 --optShapes=input:32x3x256x256 --maxShapes=input:32x3x256x256 \
    --profilingVerbosity=detailed --skipInference
```


## 5. Валидация

```bash
python validation.py --config val_config_local.yaml
```

[val_config_local.yaml](val_config_local.yaml) описывает все четыре сэтапа.

Метрики пишутся в `experiments/validation_local/validation_results.json`.

| Method | val_iou |
|---|---|
| fp16 baseline | 0.87321 |
| torch.compile | 0.87321 |
| trt fp16 | 0.87319 |
| trt int8 | 0.87298 |

## 6. Бенчмарк

Один прогон со всеми четырьмя методами через единую compute-only методику (`torch.utils.benchmark.Timer.blocked_autorange`, dummy-тензор уже на GPU, без H2D копирований внутри `__call__`):

```bash
python run_bench.py \
    --batch-size 32 --input-size 256 \
    --trt-engine "trt fp16 (polygraphy)=experiments/trt_fp16/unet_fp16.engine" \
    --trt-engine "trt int8 (polygraphy)=experiments/trt_fp16/unet_int8.engine"
```

| Method | Latency, ms | Throughput, img/s | Peak GPU, MB | Speedup |
|---|---|---|---|---|---|
| fp16 baseline | 11.25 | 2844.6 | 1623 | 1.00× |
| torch.compile | 9.87 | 3241.2 | 1506 | 1.14× |
| trt fp16 | 5.09 | 6283.3 | 108 | 2.21× |
| trt int8 | 2.96 | 10799.4 | 132 | 3.80× |


**Sanity-check — те же цифры через `trtexec --useCudaGraph`**:

```bash
trtexec --loadEngine=experiments/trt_fp16/unet_fp16.engine \
    --shapes=input:32x3x256x256 --useCudaGraph \
    --warmUp=1000 --iterations=500 --noDataTransfers
```

В нашем случае `run_bench.py` (`5.08 ms` для fp16, `2.94 ms` для int8) совпал с `trtexec --useCudaGraph` (`4.97 ms` для fp16, `2.84 ms` для int8) с разницей <3% — это подтверждает, что обёртка действительно мерит чистую compute-latency.

## 7. Sparsity 2:4 (NVIDIA Sparse Tensor Cores)

### Шаги

**1. Sparse-aware fine-tune (3 эпохи)** — на вход обычный dense ckpt, скрипт сам считает 2:4 mask, применяет, дообучает:

```bash
python3 train_sparse.py \
    --config_path train_config_sparse.yaml \
    --dense-ckpt experiments/train_local/lightning_logs/version_0/checkpoints/trained_baseline.ckpt
```

На выходе — Lightning-ckpt, в котором **ровно 49.98%** весов Conv2d/Linear = 0.0, выровненных по 2:4.

**2. Экспорт в ONNX** (тот же `export_onnx.py`):
```bash
python3 export_onnx.py \
    --ckpt experiments/train_sparse/trained_sparse_merged.ckpt \
    --onnx experiments/trt_sparse/unet_sparse.onnx
```
Физические нули переезжают из ckpt в ONNX-initializer'ы — это можно проверить:
```python
import onnx
m = onnx.load("experiments/trt_sparse/unet_sparse.onnx", load_external_data=True)
z = t = 0
for init in m.graph.initializer:
    a = onnx.numpy_helper.to_array(init)
    if a.ndim >= 2:
        t += a.size; z += int((a == 0).sum())
print(f"{100*z/t:.2f}% zeros")   # ≈ 49.98%
```

**3. Калибровка для int8-sparse-engine** — через тот же `calibrate.py`, что для dense int8:
```bash
python3 calibrate.py \
    --onnx experiments/trt_sparse/unet_sparse.onnx \
    --cache experiments/trt_sparse/calib_sparse.cache
```

**4. Build TRT engines (fp16 sparse + int8 sparse) через `trtexec --sparsity=enable`**:
```bash
# fp16 sparse
trtexec --onnx=experiments/trt_sparse/unet_sparse.onnx \
    --saveEngine=experiments/trt_sparse/unet_sparse_fp16.engine \
    --fp16 --sparsity=enable \
    --minShapes=input:1x3x256x256 --optShapes=input:32x3x256x256 --maxShapes=input:32x3x256x256 \
    --profilingVerbosity=detailed --skipInference

# int8 sparse
trtexec --onnx=experiments/trt_sparse/unet_sparse.onnx \
    --saveEngine=experiments/trt_sparse/unet_sparse_int8.engine \
    --fp16 --int8 --sparsity=enable \
    --calib=experiments/trt_sparse/calib_sparse.cache \
    --minShapes=input:1x3x256x256 --optShapes=input:32x3x256x256 --maxShapes=input:32x3x256x256 \
    --profilingVerbosity=detailed --skipInference
```

Флаг `--sparsity=enable` говорит TRT проверить, попадают ли веса в 2:4 паттерн, и **включить sparse MMA** там, где попадают.


**5. Валидация и бенчмарк** — те же `validation.py` и `run_bench.py`, добавляются 2 строки:
```bash
python3 run_bench.py --batch-size 32 --input-size 256 \
    --trt-engine "trt fp16 (polygraphy)=experiments/trt_fp16/unet_fp16.engine" \
    --trt-engine "trt int8 (polygraphy)=experiments/trt_fp16/unet_int8.engine" \
    --trt-engine "trt fp16 sparse 2:4=experiments/trt_sparse/unet_sparse_fp16.engine" \
    --trt-engine "trt int8 sparse 2:4=experiments/trt_sparse/unet_sparse_int8.engine"
```


| Method | val_iou | Latency, ms | Throughput, img/s | Peak GPU, MB | Speedup |
|---|---|---|---|---|---|
| fp16 baseline (ckpt + torch fp16) | 0.87321 | 11.25 | 2844.6 | 1623 | 1.00× |
| torch.compile | 0.87321 | 9.87 | 3241.2 | 1506 | 1.14× |
| trt fp16 (polygraphy) | 0.87319 | 5.09 | 6283.3 | 108 | 2.21× |
| trt int8 (polygraphy) | 0.87298 | 2.96 | 10799.4 | 132 | 3.80× |
| trt fp16 sparse 2:4 | 0.86832 | **4.51** | **7103.0** | 156 | **2.50×** |
| trt int8 sparse 2:4 | 0.86796 | **2.69** | **11888.2** | 180 | **4.18×** |


- **+12% к compute fp16** (`5.09 → 4.51 ms`, `2.21× → 2.50× vs baseline`).
- **+10% к compute int8** (`2.96 → 2.70 ms`, `3.80× → 4.16× vs baseline`).
- **−`5e-3` val IoU** (0.873 → 0.868) — практически в шуме fp16-арифметики, для сегментации питомцев несущественно.
- **+50 МБ peak GPU** vs dense engine — TRT хранит дополнительные sparsity-метаданные.


## 8. Визуализация графов через trex

trex рисует граф engine слой-за-слоем. [visualize_engines.py](visualize_engines.py) использует кастомный `sparse_formatter` — раскрашивает узел по precision плюс подсвечивает ярко-зелёным conv-слои, в которых TRT реально выбрал sparse Tensor Core ядро (имя тактики содержит `sparse`). Так в одной картинке видно и precision, и где именно сработало 2:4.

```bash
# экспортировать layer info + per-layer profile для всех 4-х engines
for tag in dense_fp16 dense_int8 sparse_fp16 sparse_int8; do
    trtexec --loadEngine=experiments/trex_sparse/${tag}.engine \
        --exportLayerInfo=experiments/trex_sparse/${tag}_layers.json \
        --exportProfile=experiments/trex_sparse/${tag}_profile.json \
        --shapes=input:32x3x256x256 --iterations=50 --warmUp=200 --noDataTransfers
done

# рендер PNG
python visualize_engines.py
```

Выход — `experiments/trex_sparse/unet_{dense,sparse}_{fp16,int8}.png`.

Цвета:
- **жёлтый** — FP16
- **темно зелёный** — INT8
- **красный** — FP32
- **ярко-зелёный** — conv использует sparse Tensor Core kernel.

Что видно в графах:
- **dense_fp16 / dense_int8** — ни одного ярко-зелёного блока (нет sparse-ядер, тактики из семейства `sm80_xmma_fprop_implicit_gemm_*` и `sm80_xmma_fprop_avdt_dense_int8int8_*`).
- **sparse_fp16** — **36 из 47** conv-слоёв на sparse ядрах (`sm80_xmma_fprop_sparse_conv_*_sptensor*`). Остальные 11 TRT оставил на dense kernel (в decoder'е и по краям backbone sparse-тактики проигрывают для малых shape'ов).
- **sparse_int8** — **35 из 47** conv-слоёв на `sm80_xmma_fprop_avdt_sparse_int8int8_*` и `sm80_xmma_fprop_sparse_conv_interleaved_*`.

<details>
<summary><b>Графы TRT engine (dense fp16 vs int8)</b></summary>

<table>
    <tr>
        <th align="center">dense fp16</th>
        <th align="center">dense int8</th>
    </tr>
    <tr>
        <td><img src="experiments/trex_sparse/unet_dense_fp16.png" alt="UNet TRT dense fp16" width="100%"/></td>
        <td><img src="experiments/trex_sparse/unet_dense_int8.png" alt="UNet TRT dense int8" width="100%"/></td>
    </tr>
</table>
</details>

<details>
<summary><b>Графы TRT engine (dense vs sparse, fp16)</b></summary>
<table>
<tr>
    <th align="center">dense fp16</th>
    <th align="center">sparse fp16 (2:4)</th>
</tr>
<tr>
    <td><img src="experiments/trex_sparse/unet_dense_fp16.png" alt="UNet TRT dense fp16" width="100%"/></td>
    <td><img src="experiments/trex_sparse/unet_sparse_fp16.png" alt="UNet TRT sparse fp16 2:4" width="100%"/></td>
</tr>
</table>

</details>

## Сводная таблица результатов


| Method | val_iou | Latency, ms | Throughput, img/s | Peak GPU, MB | Speedup |
|---|---|---|---|---|---|
| fp16 baseline (ckpt + torch fp16) | 0.87321 | 11.25 | 2844.6 | 1623 | 1.00× |
| torch.compile | 0.87321 | 9.87 | 3241.2 | 1506 | 1.14× |
| trt fp16 (polygraphy) | 0.87319 | 5.09 | 6283.3 | 108 | 2.21× |
| trt int8 (polygraphy) | 0.87298 | 2.96 | 10799.4 | 132 | 3.80× |
| trt fp16 sparse 2:4 | 0.86832 | **4.51** | **7103.0** | 156 | **2.50×** |
| trt int8 sparse 2:4 | 0.86796 | **2.69** | **11888.2** | 180 | **4.18×** |


- Качество **сохранено по всем методам**: разница в `val_iou` между fp16 baseline и int8 — менее `2.5e-4` IoU.
- TRT-engine быстрее baseline в **2.2× (fp16)** и **3.8× (int8)** — это совпадает с тем, что показывает `trtexec --useCudaGraph` на тех же engine (для fp16 ~4.97 ms).
- 2:4 semi-structured sparsity поверх fp16/int8 даёт ещё **+11–12%** к compute (NVIDIA Sparse Tensor Cores на A100). Качество просаживается на ~`5e-3` IoU. 
- Peak GPU memory для TRT engine — **~110 МБ**, против 1.5+ ГБ для torch (последний держит граф вычислений + lightning-overhead).


## Воспроизведение всего пайплайна одной строкой

```bash
source /datasets/unet-venv/bin/activate

CKPT=experiments/train_local/lightning_logs/version_0/checkpoints/trained_baseline.ckpt
ONNX=experiments/trt_fp16/unet.onnx
SP_CKPT=experiments/train_sparse/trained_sparse_merged.ckpt
SP_ONNX=experiments/trt_sparse/unet_sparse.onnx
SHAPES="--minShapes=input:1x3x256x256 --optShapes=input:32x3x256x256 --maxShapes=input:32x3x256x256"

# 1. dense обучение + ONNX
python train.py --config_path train_config_local.yaml && \
python export_onnx.py --ckpt $CKPT --onnx $ONNX && \

# 2. dense fp16 engine
trtexec --onnx=$ONNX --saveEngine=experiments/trt_fp16/unet_fp16.engine \
    --fp16 $SHAPES --profilingVerbosity=detailed --skipInference && \

# 3. dense int8 engine: калибровка → engine
python calibrate.py --onnx $ONNX --cache experiments/trt_fp16/calib_dense.cache && \
trtexec --onnx=$ONNX --saveEngine=experiments/trt_fp16/unet_int8.engine \
    --fp16 --int8 --calib=experiments/trt_fp16/calib_dense.cache \
    $SHAPES --profilingVerbosity=detailed --skipInference && \

# 4. sparse fine-tune + ONNX
python train_sparse.py --config_path train_config_sparse.yaml --dense-ckpt $CKPT && \
python export_onnx.py --ckpt $SP_CKPT --onnx $SP_ONNX && \

# 5. sparse fp16 engine
trtexec --onnx=$SP_ONNX --saveEngine=experiments/trt_sparse/unet_sparse_fp16.engine \
    --fp16 --sparsity=enable $SHAPES --profilingVerbosity=detailed --skipInference && \

# 6. sparse int8 engine: калибровка → engine
python calibrate.py --onnx $SP_ONNX --cache experiments/trt_sparse/calib_sparse.cache && \
trtexec --onnx=$SP_ONNX --saveEngine=experiments/trt_sparse/unet_sparse_int8.engine \
    --fp16 --int8 --sparsity=enable --calib=experiments/trt_sparse/calib_sparse.cache \
    $SHAPES --profilingVerbosity=detailed --skipInference && \

# 7. validation + bench всех 6 методов
python validation.py --config val_config_local.yaml && \
python validation.py --config val_config_sparse.yaml && \
python run_bench.py --batch-size 32 --input-size 256 \
    --trt-engine "trt fp16 (polygraphy)=experiments/trt_fp16/unet_fp16.engine" \
    --trt-engine "trt int8 (polygraphy)=experiments/trt_fp16/unet_int8.engine" \
    --trt-engine "trt fp16 sparse 2:4=experiments/trt_sparse/unet_sparse_fp16.engine" \
    --trt-engine "trt int8 sparse 2:4=experiments/trt_sparse/unet_sparse_int8.engine"
```

## Docker

В образе уже установлено всё: код пайплайна, зависимости и веса, движки.


Сборка образа:

```bash
docker build -t unet-infer .
```

Запуск:

```bash
docker run --rm -it --gpus all unet-infer
```

Внутри контейнера сразу доступны команды из соответствующих секций README:

```bash
# бенчмарк всех 4 методов на готовых engine
python3 run_bench.py --batch-size 32 --input-size 256 \
    --trt-engine "trt fp16 (polygraphy)=experiments/trt_fp16/unet_fp16.engine" \
    --trt-engine "trt int8 (polygraphy)=experiments/trt_fp16/unet_int8.engine"

# валидация (метрики IoU)
python3 validation.py --config val_config_local.yaml

# демо-инференс на реальных картинках
python3 inference_demo.py \
    --ckpt experiments/train_local/lightning_logs/version_0/checkpoints/trained_baseline.ckpt \
    --out /tmp/demo.png

# повторное обучение с нуля
python3 train.py --config_path train_config_local.yaml
```

Артефакты в `experiments/` (чекпоинт, ONNX, оба `.engine`, `calib.cache`, графы).

Адрес образа
TODO
