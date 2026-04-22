# Не получилось заставить это работать на A100, хотя на v100 работало на cpu (и квантизация на cpu и инференс на cpu)

# import argparse
# import yaml
# import os
# import torch
# import numpy as np
# import onnxruntime as ort
# import segmentation_models_pytorch as smp
# import modelopt.torch.quantization as mtq # Тот самый второй метод
# from train import SegmentationModule, get_loaders
# from pathlib import Path


# import os
# os.environ["TORCH_EXPORT_DECOMPOSE_NON_STRICT"] = "1"
# # Also try:
# os.environ["TORCHDYNAMO_DISABLE"] = "1" 


# def print_optimization_dependencies():
#     import torch
#     import modelopt
#     try:
#         import tensorrt as trt
#         trt_version = trt.__version__
#     except ImportError:
#         trt_version = "Not Installed"
    
#     try:
#         import onnxruntime as ort
#         ort_version = ort.__version__
#     except ImportError:
#         ort_version = "Not Installed"
    
#     print("=" * 60)
#     print("CRITICAL DEPENDENCIES FOR INT8 QUANTIZATION")
#     print("=" * 60)
#     print(f"{'Package':<20} | {'Version':<20}")
#     print("-" * 60)
#     print(f"{'PyTorch':<20} | {torch.__version__}")
#     print(f"{'CUDA Available':<20} | {torch.cuda.is_available()}")
#     print(f"{'NVIDIA ModelOpt':<20} | {modelopt.__version__}")
#     print(f"{'TensorRT (System)':<20} | {trt_version}")
#     print(f"{'ONNX Runtime':<20} | {ort_version}")
#     print("=" * 60)
#     print()

# def main(config):
#     # 1. Загрузка исходной модели
#     model_module = SegmentationModule.load_from_checkpoint(
#         checkpoint_path=config["model_path"],
#         model=smp.Unet(**config["model"])
#     )
#     model = model_module.model.eval().to(torch.device("cpu"))

#     # 2. Подготовка данных для калибровки
#     _, test_loader = get_loaders(**config["data"])

#     # Функция-прогон для калибратора
#     def calibrate_loop(model):
#         print("Запуск калибровки внутри PyTorch...")
#         with torch.no_grad():
#             for i, batch in enumerate(test_loader):
#                 if i >= 2: break # 10 батчей обычно хватает
#                 # batch[0] - это картинки
#                 model(batch[0])

#     # 3. Квантование (PTQ) через Modelopt
#     # Это создаст QDQ-версию модели прямо в PyTorch
#     print("Применяем квантование Modelopt (INT8)...")
#     quantized_model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate_loop)
    
#     # 4. Экспорт в ONNX
#     # Теперь ONNX будет содержать узлы QuantizeLinear/DequantizeLinear (QDQ)
#     onnx_path = Path(config["exp_dir"]) / "unet_int8_qdq.onnx"
#     os.makedirs(onnx_path.parent, exist_ok=True)
    
#     quantized_model.cpu()
#     dummy_input_cpu = torch.randn(1, 3, 256, 256) # Без .cuda()

#     torch.onnx.export(
#         quantized_model,
#         dummy_input_cpu,
#         str(onnx_path),
#         export_params=True,
#         opset_version=14, 
#         do_constant_folding=True,
#         input_names=['input'],
#         output_names=['output'],
#         # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
#     )


# def test_model(model_path):
#     # Путь к файлу
#     if not Path(model_path).exists():
#         print(f"Файл {model_path} не найден!")
#         return

#     dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    
#     # 1. Тест на CPU
#     print("\n--- Проверка на CPU ---")
#     try:
#         sess_cpu = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
#         input_name = sess_cpu.get_inputs()[0].name
#         res_cpu = sess_cpu.run(None, {input_name: dummy_input})
#         print(f"Успех! Выход получен, форма: {res_cpu[0].shape}")
#     except Exception as e:
#         print(f"Ошибка на CPU: {e}")

#     # 2. Тест на GPU
#     if torch.cuda.is_available():
#         print("\n--- Проверка на GPU (TensorRT) ---")
#         # На V100 здесь может быть ошибка SM 70, что подтвердит готовность кода для A100
#         providers = [
#             ('TensorrtExecutionProvider', {
#                 'trt_fp16_enable': True,
#                 'trt_int8_enable': True,
#             }),
#             'CUDAExecutionProvider'
#         ]
#         try:
#             sess_gpu = ort.InferenceSession(model_path, providers=providers)
#             input_name = sess_gpu.get_inputs()[0].name
#             res_gpu = sess_gpu.run(None, {input_name: dummy_input})
#             print(f"Успех! Выход получен, форма: {res_gpu[0].shape}")
#             print(f"Активный провайдер: {sess_gpu.get_providers()[0]}")
#         except Exception as e:
#             print(f"Результат на текущем GPU: {e}")
#             print("Это ожидаемо для V100. Главное, что CPU тест прошел — значит файл корректен.")



# if __name__ == "__main__":
#     print_optimization_dependencies()
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config.yaml")
#     args = parser.parse_args()

#     with open(args.config, "r") as f:
#         config = yaml.safe_load(f)
#     main(config)
#     onnx_path = Path(config["exp_dir"]) / "unet_int8_qdq.onnx"
#     test_model(str(onnx_path))
