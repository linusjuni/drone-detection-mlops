FROM ghcr.io/astral-sh/uv:0.5.21 AS uv_bin

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY --from=uv_bin /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

COPY src/drone_detector_mlops/ ./drone_detector_mlops/

ENV PYTHONPATH=/app

RUN ls -R /app

RUN uv pip install --system --no-cache .

ENTRYPOINT ["python", "-m", "drone_detector_mlops.workflows.train"]
