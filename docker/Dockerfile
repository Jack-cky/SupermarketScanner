ARG PYTHON_IMAGE=python:3.10.12-slim-buster

# builder stage
FROM ${PYTHON_IMAGE} AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --target=packages -r requirements.txt

FROM ${PYTHON_IMAGE}

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# final stage
RUN useradd -m viewer

WORKDIR /supermarketscanner

RUN mkdir -p runs && chown -R viewer:viewer runs

COPY --from=builder /app/packages /usr/local/lib/python3.10/site-packages

COPY demo demo
COPY models/fine_tune.onnx models/fine_tune.onnx
COPY supermarketscanner supermarketscanner

ENV PATH="/usr/local/lib/python3.10/site-packages/bin:$PATH"

USER viewer

CMD ["streamlit", "run", "demo/streamlit_app.py", "--server.fileWatcherType=none", "--server.address=0.0.0.0"]
