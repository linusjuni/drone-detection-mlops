FROM ghcr.io/astral-sh/uv:0.5.21 AS uv_bin

FROM python:3.12-slim

WORKDIR /app

COPY --from=uv_bin /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

COPY src/drone_detector_mlops/ ./drone_detector_mlops/

ENV PYTHONPATH=/app

RUN uv pip install --system --no-cache .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "drone_detector_mlops.api.main:app"]

CMD ["--host", "0.0.0.0", "--port", "8000"]
