from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from datetime import datetime, timezone

from drone_detector_mlops.api.inference import predict_image, load_model_singleton
from drone_detector_mlops.api.schemas import PredictionResponse, HealthResponse, InfoResponse
from drone_detector_mlops.utils.settings import settings
from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Drone Detector API",
    description="Drone vs Bird classification API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time for uptime
startup_time = datetime.now(timezone.utc)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("API starting up")
    load_model_singleton()
    logger.success("API ready")


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

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB")

    # Open image and convert to RGB (handles RGBA/grayscale)
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Run prediction
    try:
        prediction, metadata = predict_image(image)

        logger.info(
            "Prediction complete",
            class_name=prediction.class_name,
            confidence=prediction.confidence,
            time_ms=metadata.inference_time_ms,
        )

        return PredictionResponse(prediction=prediction, metadata=metadata)

    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
