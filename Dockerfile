FROM nvcr-proxy.kontur.host/nvidia/tritonserver:26.03-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git graphviz && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скрипты пайплайна + локальные конфиги
COPY *.py *.yaml ./

# Артефакты эксперимента: ckpt, ONNX, TRT engines (fp16/int8), калибровочный cache, графы.
# Кладутся в experiments/ отдельным шагом перед docker build — см. README.
COPY experiments/ ./experiments/

CMD ["bash"]
