import datetime
import torch
from PIL import Image
import time

from drone_detector_mlops.api.schemas import Prediction, PredictionMetadata, PredictionScores
from drone_detector_mlops.model import get_model
from drone_detector_mlops.utils.storage import get_storage
from drone_detector_mlops.utils.settings import settings
from drone_detector_mlops.data.transforms import test_transform
from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)

# Model singleton
_model_cache = None
_model_version = None


def load_model_singleton():
    """Load model once and cache it."""
    global _model_cache, _model_version

    if _model_cache is None:
        logger.info("Loading model", filename=settings.MODEL_FILENAME)
        storage = get_storage()

        # Load state dict
        state_dict = storage.load_model(settings.MODEL_FILENAME)

        # Create model and load weights
        model = get_model(pretrained=False)
        model.load_state_dict(state_dict)
        model.eval()  # Set to eval mode

        _model_cache = model
        _model_version = settings.MODEL_FILENAME
        logger.success("Model loaded successfully")

    return _model_cache


def predict_image(image: Image.Image) -> tuple[Prediction, PredictionMetadata]:
    """Run inference on a single image."""
    start_time = time.time()

    # Load model
    model = load_model_singleton()

    # Transform image
    image_tensor = test_transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Extract results
    confidence, predicted_class = torch.max(probs, 0)
    class_names = ["drone", "bird"]

    inference_time = (time.time() - start_time) * 1000

    # Create Pydantic objects
    prediction = Prediction(
        class_name=class_names[predicted_class.item()],
        confidence=confidence.item(),
        scores=PredictionScores(
            drone=probs[0].item(),
            bird=probs[1].item(),
        ),
    )

    metadata = PredictionMetadata(
        model_version=_model_version,
        inference_time_ms=inference_time,
        timestamp=datetime.utcnow(),
    )

    return prediction, metadata
