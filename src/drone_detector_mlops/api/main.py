import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from drone_detector_mlops.api.inference import load_model_singleton, predict_image
from drone_detector_mlops.api.schemas import HealthResponse, InfoResponse, PredictionResponse
from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.utils.settings import settings

logger = get_logger(__name__)

# Track startup time for uptime
startup_time: datetime | None = None

# Prometheus metrics
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

predictions_by_class_total = Counter(
    "predictions_by_class_total",
    "Total predictions by predicted class",
    ["class_name"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction inference time in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

request_size_bytes = Histogram(
    "request_size_bytes",
    "Size of uploaded images in bytes",
    buckets=[10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
)

http_errors_total = Counter(
    "http_errors_total",
    "Total HTTP errors by status code and reason",
    ["status_code", "reason"],
)

model_loaded_info = Gauge(
    "model_loaded_info",
    "Model loading status with version label (1=loaded)",
    ["model_version"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global startup_time

    # Startup
    logger.info("API starting up")
    load_model_singleton()
    startup_time = datetime.now(timezone.utc)

    # Set model loaded gauge with version label
    from drone_detector_mlops.api.inference import _model_version

    model_loaded_info.labels(model_version=_model_version or "unknown").set(1)
    logger.success("API ready")

    yield

    # Shutdown
    model_loaded_info.labels(model_version=_model_version or "unknown").set(0)
    logger.info("API shutting down")


# Create FastAPI app
app = FastAPI(
    title="Drone Detector API",
    description="Drone vs Bird classification API",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from drone_detector_mlops.api.inference import _model_cache

    return HealthResponse(status="healthy", model_loaded=_model_cache is not None)


@app.get("/v1/info", response_model=InfoResponse)
async def get_info():
    """Get API information."""
    from drone_detector_mlops.api.inference import _model_version

    uptime = (datetime.now(timezone.utc) - startup_time).total_seconds()

    return InfoResponse(model_version=_model_version or "unknown", uptime_seconds=uptime)


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict drone or bird from uploaded image."""
    prediction_requests_total.inc()

    # Validate file type
    if not file.content_type.startswith("image/"):
        http_errors_total.labels(status_code="400", reason="invalid_content_type").inc()
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and validate file size
    contents = await file.read()
    request_size_bytes.observe(len(contents))
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        http_errors_total.labels(status_code="413", reason="file_too_large").inc()
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB")

    # Open image and convert to RGB (handles RGBA/grayscale)
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        http_errors_total.labels(status_code="400", reason="invalid_image").inc()
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Run prediction
    try:
        prediction, metadata = predict_image(image)

        # Record metrics
        prediction_latency_seconds.observe(metadata.inference_time_ms / 1000)
        predictions_by_class_total.labels(class_name=prediction.class_name).inc()

        logger.info(
            "Prediction complete",
            class_name=prediction.class_name,
            confidence=prediction.confidence,
            time_ms=metadata.inference_time_ms,
        )

        return PredictionResponse(prediction=prediction, metadata=metadata)

    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        http_errors_total.labels(status_code="500", reason="prediction_failed").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
