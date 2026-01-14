from datetime import datetime, timezone
import numpy as np
import onnxruntime as ort
from PIL import Image

from drone_detector_mlops.api.schemas import Prediction, PredictionMetadata, PredictionScores
from drone_detector_mlops.utils.storage import get_storage
from drone_detector_mlops.utils.settings import settings
from drone_detector_mlops.data.transforms import test_transform
from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)

# Model singleton
_model_cache = None
_model_version = None


def load_model_singleton():
    """Load ONNX model once and cache it."""
    global _model_cache, _model_version

    if _model_cache is None:
        logger.info("Loading ONNX model", filename=settings.MODEL_FILENAME)
        storage = get_storage()

        # Load ONNX model path
        model_path = storage.load_onnx_path(settings.MODEL_FILENAME)

        # Create ONNX Runtime session
        _model_cache = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _model_version = settings.MODEL_FILENAME
        logger.success("ONNX model loaded successfully", path=str(model_path))

    return _model_cache


def predict_image(image: Image.Image) -> tuple[Prediction, PredictionMetadata]:
    """Run inference on a single image using ONNX Runtime."""
    start_time = datetime.now(timezone.utc)

    # Load model
    model = load_model_singleton()

    # Transform image (same as before)
    image_tensor = test_transform(image).unsqueeze(0)

    # Convert to numpy for ONNX Runtime
    input_array = image_tensor.numpy()

    # Run inference with ONNX Runtime
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    logits = model.run([output_name], {input_name: input_array})[0]

    # Convert logits to probabilities (softmax)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs = probs[0]  # Remove batch dimension

    # Extract results (same logic as before)
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    class_names = ["drone", "bird"]

    inference_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

    # Create Pydantic objects (same as before)
    prediction = Prediction(
        class_name=class_names[predicted_class],
        confidence=confidence,
        scores=PredictionScores(
            drone=float(probs[0]),
            bird=float(probs[1]),
        ),
    )

    metadata = PredictionMetadata(
        model_version=_model_version,
        inference_time_ms=inference_time,
        timestamp=datetime.now(timezone.utc),
    )

    return prediction, metadata
