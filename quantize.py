import argparse
import yaml
import os
os.environ["LD_LIBRARY_PATH"] += ":/root/efficient_dl/efficient_dl_env/lib/python3.11/site-packages/tensorrt_libs"
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
from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibrationDataReader


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

    exp_dir = Path(config["exp_dir"])
    onnx_path = exp_dir / "unet_model.onnx"
    # Для TensorRT важно расширение .flatbuffers или .cache
    calibration_table = str(exp_dir / "calibration.flatbuffers") 
    cache_dir = str(exp_dir / "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    # torch.onnx.export(
    #     model_module.model,
    #     dummy_input, 
    #     str(onnx_path), 
    #     export_params=True, 
    #     opset_version=17,
    #     do_constant_folding=True, 
    #     input_names=['input'], 
    #     output_names=['output'],
    #     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    #     # Это заставит использовать классический скриптовый экспортер
    #     operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH 
    # )
    # torch.onnx.export(model_module.model, dummy_input, str(onnx_path), verbose=False, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    # 1. Экспорт в стандартный ONNX (без квантования внутри)
    model_module.eval().cpu() # Экспортировать лучше на CPU для стабильности
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model_module.model, 
        dummy_input, 
        str(onnx_path), 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=13, # Стабильная версия для TRT
        do_constant_folding=True
    )
    print("Модель сконвертирована в ONNX!")

    
    print("Начало калибровки...")
    _, test_loader = get_loaders(input_size=config["data"]["input_size"], batch_size=1, num_workers=config["data"]["num_workers"])
    data_reader = UNetDataReader(test_loader, input_name='input')
    
    # Создаем калибратор. Он создаст "augmented_model.onnx" во временной папке
    augmented_model_path = str(exp_dir / "augmented_model.onnx")
    calibrator = create_calibrator(str(onnx_path), [], augmented_model_path=augmented_model_path, providers=["CUDAExecutionProvider"])
    # calibrator.set_execution_providers(["CUDAExecutionProvider"]) 

    for i in range(2):
        data = data_reader.get_next()
        if data is None: break
        calibrator.collect_data(data_reader) # Внутри он вызовет get_next
    
    write_calibration_table(calibrator.compute_range(), calibration_table)
    print(f"Таблица калибровки сохранена: {calibration_table}")

    # 3. ЗАПУСК ИНФЕРЕНСА С ТАБЛИЦЕЙ
    print("Запуск TensorRT сессии...")
    providers = [
        ('TensorrtExecutionProvider', {
            'trt_fp16_enable': True,
            'trt_int8_enable': True,
            'trt_int8_calibration_table_name': calibration_table,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
        }),
        'CUDAExecutionProvider'
    ]

    # Используем ОРИГИНАЛЬНЫЙ onnx_path, а не квантованный! 
    # TRT сам подтянет таблицу калибровки.
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    print("Запуск сессии")
    
    # Тестовый прогон
    test_batch = next(iter(test_loader))[0].numpy()
    outputs = session.run(None, {'input': test_batch})
    print("Успех! Сессия создана, тензор получен.")

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
