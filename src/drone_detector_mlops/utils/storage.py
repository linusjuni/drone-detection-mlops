from pathlib import Path
from typing import Union
import torch
import tempfile
from google.cloud import storage as gcs

from drone_detector_mlops.utils.settings import settings
from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class StorageContext:
    """Unified storage interface"""

    def __init__(self, mode: str):
        self.mode = mode
        self._gcs_client = None

    @property
    def data_dir(self) -> Union[Path, str]:
        """Returns data directory path (local or GCS)."""
        if self.mode == "cloud":
            return settings.GCS_DATA_PATH
        return Path("data")

    @property
    def splits_dir(self) -> Union[Path, str]:
        """Returns splits directory path (local or GCS)."""
        if self.mode == "cloud":
            return f"{settings.GCS_DATA_PATH}/splits"
        return Path("data/splits")

    @property
    def models_dir(self) -> Union[Path, str]:
        """Returns models directory path (local or GCS)."""
        if self.mode == "cloud":
            return f"{settings.GCS_MODELS_BUCKET}/checkpoints"
        return Path("models")

    @property
    def gcs_client(self):
        """Lazy-load GCS client only when needed."""
        if self._gcs_client is None and self.mode == "cloud":
            self._gcs_client = gcs.Client(project=settings.GCP_PROJECT)
        return self._gcs_client

    def save_model(self, state_dict: dict, filename: str) -> Union[Path, str]:
        """Save model checkpoint."""
        if self.mode == "local":
            return self._save_local(state_dict, filename)
        else:
            return self._save_cloud(state_dict, filename)

    def _save_local(self, state_dict: dict, filename: str) -> Path:
        """Save model to local filesystem."""
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / filename
        torch.save(state_dict, model_path)
        logger.success("Model saved locally", path=str(model_path))
        return model_path

    def _save_cloud(self, state_dict: dict, filename: str) -> str:
        """Save model directly to GCS (no local copy)."""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
            tmp_path = tmp_file.name
            torch.save(state_dict, tmp_path)

        # Upload to GCS
        gcs_path = f"{settings.GCS_MODELS_BUCKET}/checkpoints/{filename}"
        bucket_name = settings.GCS_MODELS_BUCKET.replace("gs://", "").split("/")[0]
        blob_path = f"checkpoints/{filename}"

        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(tmp_path)

        # Clean up temp file
        Path(tmp_path).unlink()

        logger.success("Model saved to GCS", path=gcs_path)
        return gcs_path


def get_storage() -> StorageContext:
    """Factory function to get storage context based on mode."""
    mode = settings.MODE
    if mode not in ["local", "cloud"]:
        raise ValueError(f"Invalid MODE: {mode}. Must be 'local' or 'cloud'")

    logger.info("Storage context initialized", mode=mode)
    return StorageContext(mode=mode)
