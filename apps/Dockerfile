FROM python:3.12.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY checkpoints/smktscnr_ft.onnx ./checkpoints/smktscnr_ft.onnx
COPY config/products.yaml ./config/products.yaml
COPY smktscnr ./smktscnr
COPY pyproject.toml ./

FROM base AS kiosk

COPY apps/kiosk/main.py ./apps/kiosk/main.py

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install \
        --index-url ${TORCH_INDEX_URL} \
        --extra-index-url https://pypi.org/simple \
        torch \
        torchvision && \
    pip install --no-cache-dir --no-compile . && \
    useradd --create-home smktscnr && \
    chown -R smktscnr:smktscnr /app

USER smktscnr

CMD ["python", "-m", "apps.kiosk.main"]

FROM base AS demo

COPY apps/demo/assets ./apps/demo/assets
COPY apps/demo/main.py ./apps/demo/main.py

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install \
        --index-url ${TORCH_INDEX_URL} \
        --extra-index-url https://pypi.org/simple \
        torch \
        torchvision && \
    pip install --no-cache-dir --no-compile '.[demo]' && \
    useradd --create-home smktscnr && \
    chown -R smktscnr:smktscnr /app

USER smktscnr

EXPOSE 8501

CMD ["streamlit", "run", "apps/demo/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
