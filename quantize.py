import torch


class TRTModelWrapper:
    """Оборачивает готовый TRT engine (polygraphy TrtRunner), эмулируя forward nn.Module.

    Используется в validation.py: путь до .engine → объект, у которого есть .eval()
    и .__call__(torch.Tensor) -> torch.Tensor. Внутри numpy/CPU-копирование на каждый вызов
    (для compute-only бенчмарка см. TRTComputeRunner в run_bench.py).
    """

    def __init__(self, engine_path, input_name="input", output_name="output"):
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        self._np = np
        self._input_name = input_name
        self._output_name = output_name
        self._runner = TrtRunner(EngineFromBytes(BytesFromPath(engine_path)))
        self._activated = False

    def _ensure_active(self):
        if not self._activated:
            self._runner.activate()
            self._activated = True

    def __call__(self, x):
        self._ensure_active()
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else self._np.asarray(x)
        x_np = self._np.ascontiguousarray(x_np, dtype=self._np.float32)
        outputs = self._runner.infer(feed_dict={self._input_name: x_np})
        return torch.from_numpy(outputs[self._output_name]).to(device)

    def eval(self):
        return self

    def close(self):
        if self._activated:
            self._runner.deactivate()
            self._activated = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
