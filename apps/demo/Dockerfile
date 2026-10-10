FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

WORKDIR /app

COPY config/products.yaml ./config/products.yaml
COPY smktscnr ./smktscnr
COPY pyproject.toml ./
COPY apps/demo/assets ./apps/demo/assets
COPY apps/demo/main.py ./apps/demo/main.py

RUN --mount=from=ghcr.io/astral-sh/uv:0.10.11,source=/uv,target=/bin/uv \
    echo opencv-python | uv pip install --excludes - '.[demo,headless]' && \
    useradd --create-home smktscnr && \
    chown -R smktscnr:smktscnr /app

USER smktscnr

EXPOSE 8501

CMD ["streamlit", "run", "apps/demo/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
