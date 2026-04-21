import argparse
import yaml
import os
import json
import joblib
import torch
import lightning as L
import segmentation_models_pytorch as smp
import onnxruntime as ort
import tensorrt
import numpy as np
import time
from train import SegmentationModule, get_loaders
from pathlib import Path
from onnxruntime.quantization import (
    CalibrationDataReader, write_calibration_table,
    calibrate, quantize_static, QuantType, QuantFormat, CalibrationMethod
)


class UNetDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name, limit=None):
        self.dataloader = dataloader
        self.input_name = input_name
        self.limit = limit  # Лимит объектов
        self.counter = 0    # Текущий счетчик
        self.dataset_iter = iter(dataloader)

    def get_next(self):
        # Если лимит задан и мы его достигли — останавливаемся
        if self.limit is not None and self.counter >= self.limit:
            return None

        batch = next(self.dataset_iter, None)
        if batch is None:
            return None

        self.counter += 1
        # Берем только картинки без маски
        return {self.input_name: batch[0].numpy()}
    
    def __len__(self):
        # Если лимит задан, возвращаем его, иначе длину лоадера
        if self.limit is not None:
            return min(self.limit, len(self.dataloader))
        return len(self.dataloader)
    
    def rewind(self):
        """Сброс итератора и счетчика в начало"""
        self.dataset_iter = iter(self.dataloader)
        self.counter = 0


class TRTModelWrapper:
    def __init__(self, onnx_path, calibration_path, cache_dir):
        self.onnx_path = onnx_path
        self.calibration_path = calibration_path
        self.cache_dir = cache_dir
        self.session = None

    def __call__(self, x):
        if hasattr(x, "device"):
            x = x.detach().cpu().numpy()

        if self.session is None:
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_fp16_enable': True,
                    'trt_int8_enable': True,
                    'trt_int8_calibration_table_name': self.calibration_path,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': self.cache_dir
                }),
                'CUDAExecutionProvider'
            ]
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        outputs = self.session.run(None, {'input': x})
        return torch.from_numpy(outputs[0])

    def eval(self):
        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    model_module = SegmentationModule.load_from_checkpoint(
        checkpoint_path=config["model_path"],
        model=smp.Unet(**config["model"])
    )
    model_module.eval().cuda()
    dummy_input = torch.randn(1, 3, 256, 256).cuda()

    onnx_path = Path(config["exp_dir"]) / "unet_model.onnx"
    calibration_path = Path(config["exp_dir"]) / "calibration.cache"
    cache_dir = Path(config["exp_dir"]) / "cache_dir"
    os.makedirs(cache_dir, exist_ok=True)

    torch.onnx.export(
        model_module.model,
        dummy_input, 
        str(onnx_path), 
        export_params=True, 
        opset_version=14,
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Модель сконвертирована в ONNX!")

    # if not os.path.exists(calibration_table):
    #     open(calibration_table, 'a').close()

    providers = [
        ('TensorrtExecutionProvider', {
            'trt_fp16_enable': True,
            'trt_int8_enable': True,
            # 'trt_int8_calibration_table_name': calibration_path,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(cache_dir),
            'trt_int8_use_native_calibration_table': False, 
        }),
        'CUDAExecutionProvider'
    ]

    print("Инициализация сессии TensorRT (может занять время)...")
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    print("Запуск сессии")
    
    # 3. Чтобы калибровка реально произошла, нужно прогнать данные
    _, test_loader = get_loaders(**config["data"])
    for i, batch in enumerate(test_loader):
        if i > 2: break # Хватит 100 батчей для калибровки
        input_data = batch[0].numpy()
        session.run(None, {'input': input_data})
    
    print("Вроде работает")
    
    # print(f"Калибровка завершена. Файл сохранен в: {calibration_path}")

def print_env_info():
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  CUDA (runtime) : {torch.version.cuda}")
    print(f"  cuDNN          : {torch.backends.cudnn.version()}")
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  ort            : {ort.__version__}")
    print(f"  tensorrt       : {tensorrt.__version__}")
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/путь/к/вашему/env/lib/python3.11/site-packages/tensorrt_libs
    print(f'LD_LIBRARY_PATH {os.environ.get("LD_LIBRARY_PATH", "LD_LIBRARY_PATH пустой")}')
    print("=" * 60)
    print()

if __name__ == "__main__":
    print_env_info()
    main()
