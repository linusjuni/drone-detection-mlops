# Drone Detector API

FastAPI inference endpoint for drone vs bird classification.

## Quick Start

### 1. Run the API locally

```bash
MODE=local uv run uvicorn drone_detector_mlops.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test with an image

```bash
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@data/drone/1.JPEG"
```

### 3. Or use interactive docs

Open `http://localhost:8000/docs` in your browser and try the endpoints

## Endpoints

- **`GET /health`** - Check if API is running
- **`GET /v1/info`** - Get model version and uptime
- **`POST /v1/predict`** - Upload image, get prediction

## How It Works

1. **Model Loading**: On startup, the API loads `models/model-latest.onnx` using ONNX Runtime
2. **Image Processing**: Uploaded images are transformed using the same preprocessing as training
3. **Inference**: ONNX Runtime predicts drone or bird with confidence scores
4. **Response**: Returns structured JSON with prediction + metadata

## Storage Modes

The API uses the same storage abstraction as training:

- **Local mode** (`MODE=local`): Loads from `models/model-latest.onnx`
- **Cloud mode** (`MODE=cloud`): Loads from GCS bucket (`gs://drone-detection-mlops-models/checkpoints/model-latest.onnx`)

Training still uses PyTorch - we export to ONNX automatically during model saving.
