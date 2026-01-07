FROM ghcr.io/astral-sh/uv:0.5.21 AS uv_bin

FROM python:3.12-slim

WORKDIR /app

COPY --from=uv_bin /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv pip install --system -r pyproject.toml

COPY src/ /app/src/

RUN uv pip install --system -e .

COPY data/splits/ /app/data/splits/

ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "-m", "drone_detector_mlops.workflows.train"]

CMD ["--data-dir", "data", "--output-dir", "models", "--epochs", "10", "--batch-size", "32", "--lr", "0.001"]
# EOF
