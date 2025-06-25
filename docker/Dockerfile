ARG PYTHON_IMAGE=python:3.10.12-slim-bullseye

# builder stage
FROM ${PYTHON_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY supermarketscanner/ supermarketscanner/
COPY requirements.txt pyproject.toml setup.py ./

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/venv/bin/pip install . --no-deps

# final stage
FROM ${PYTHON_IMAGE}

WORKDIR /supermarketscanner

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY demo/ demo/
COPY models/fine_tune.onnx models/fine_tune.onnx
COPY supermarketscanner/ supermarketscanner/

# execution
RUN useradd -m viewer && \
    chown -R viewer:viewer /supermarketscanner

USER viewer

EXPOSE 8501

CMD ["streamlit", "run", "demo/streamlit_app.py", "--server.fileWatcherType=none", "--server.address=0.0.0.0"]
