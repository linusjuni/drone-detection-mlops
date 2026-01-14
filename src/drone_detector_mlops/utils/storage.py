from pathlib import Path
from typing import Union
import gcsfs
import torch
import onnx
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

    def save_model(self, model: torch.nn.Module, filename: str) -> Union[Path, str]:
        """Save model checkpoint in both PyTorch and ONNX formats.

        Creates 4 files:
        - {filename}.pth (timestamped PyTorch)
        - {filename}.onnx (timestamped ONNX)
        - model-latest.pth (always current PyTorch)
        - model-latest.onnx (always current ONNX)
        """
        if self.mode == "local":
            return self._save_local(model, filename)
        else:
            return self._save_cloud(model, filename)

    def _save_local(self, model: torch.nn.Module, filename: str) -> Path:
        """Save model to local filesystem (PyTorch + ONNX)."""
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Extract state dict
        state_dict = model.state_dict()

        # Save timestamped PyTorch model
        pth_path = models_dir / f"{filename}.pth"
        torch.save(state_dict, pth_path)
        logger.success("PyTorch model saved locally", path=str(pth_path))

        # Save latest PyTorch model
        latest_pth_path = models_dir / "model-latest.pth"
        torch.save(state_dict, latest_pth_path)
        logger.success("Latest PyTorch model updated", path=str(latest_pth_path))

        # Convert and save timestamped ONNX model
        onnx_path = models_dir / f"{filename}.onnx"
        self._convert_to_onnx(model, onnx_path)
        logger.success("ONNX model saved locally", path=str(onnx_path))

        # Save latest ONNX model
        latest_onnx_path = models_dir / "model-latest.onnx"
        self._convert_to_onnx(model, latest_onnx_path)
        logger.success("Latest ONNX model updated", path=str(latest_onnx_path))

        return pth_path

    def _save_cloud(self, model: torch.nn.Module, filename: str) -> str:
        """Save model to GCS (PyTorch + ONNX)."""
        bucket_name = settings.GCS_MODELS_BUCKET.replace("gs://", "").split("/")[0]
        bucket = self.gcs_client.bucket(bucket_name)

        # Extract state dict
        state_dict = model.state_dict()

        # Save timestamped PyTorch model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
            tmp_path = tmp_file.name
            torch.save(state_dict, tmp_path)

        pth_blob_path = f"checkpoints/{filename}.pth"
        blob = bucket.blob(pth_blob_path)
        blob.upload_from_filename(tmp_path)
        Path(tmp_path).unlink()
        logger.success("PyTorch model saved to GCS", path=f"gs://{bucket_name}/{pth_blob_path}")

        # Save latest PyTorch model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
            tmp_path = tmp_file.name
            torch.save(state_dict, tmp_path)

        latest_pth_blob = bucket.blob("checkpoints/model-latest.pth")
        latest_pth_blob.upload_from_filename(tmp_path)
        Path(tmp_path).unlink()
        logger.success("Latest PyTorch model updated in GCS")

        # Save timestamped ONNX model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_file:
            tmp_onnx_path = tmp_file.name
            self._convert_to_onnx(model, tmp_onnx_path)

        onnx_blob_path = f"checkpoints/{filename}.onnx"
        onnx_blob = bucket.blob(onnx_blob_path)
        onnx_blob.upload_from_filename(tmp_onnx_path)
        Path(tmp_onnx_path).unlink()
        logger.success("ONNX model saved to GCS", path=f"gs://{bucket_name}/{onnx_blob_path}")

        # Save latest ONNX model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_file:
            tmp_onnx_path = tmp_file.name
            self._convert_to_onnx(model, tmp_onnx_path)

        latest_onnx_blob = bucket.blob("checkpoints/model-latest.onnx")
        latest_onnx_blob.upload_from_filename(tmp_onnx_path)
        Path(tmp_onnx_path).unlink()
        logger.success("Latest ONNX model updated in GCS")

        return f"gs://{bucket_name}/{pth_blob_path}"

    def _convert_to_onnx(self, model: torch.nn.Module, output_path: Union[str, Path]) -> None:
        """Convert PyTorch model to ONNX format.

        Args:
            model: PyTorch model to convert
            output_path: Path where ONNX model will be saved
        """
        # Ensure model is on CPU and in eval mode
        model = model.cpu()
        model.eval()

        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
                verbose=False,
                dynamo=False,
            )

        # Validate ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Log model size for verification
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model size: {file_size_mb:.2f} MB")

    def load_state_dict(self, filename: str):
        """Load model state dict from storage (local or GCS)."""
        if self.mode == "local":
            model_path = self.models_dir / filename
            return torch.load(model_path, map_location="cpu")
        else:  # cloud mode
            model_path = f"{self.models_dir}/{filename}"
            fs = gcsfs.GCSFileSystem()
            with fs.open(model_path, "rb") as f:
                return torch.load(f, map_location="cpu")

    def load_onnx_path(self, filename: str = "model-latest.onnx") -> Union[Path, str]:
        """Load ONNX model path from storage (local or GCS)."""
        if self.mode == "local":
            model_path = self.models_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            return model_path
        else:  # cloud mode
            # Download from GCS to temp location
            model_path = f"{self.models_dir}/{filename}"
            fs = gcsfs.GCSFileSystem()

            # Save to temp file
            temp_path = Path(tempfile.gettempdir()) / filename
            with fs.open(model_path, "rb") as f_in:
                with open(temp_path, "wb") as f_out:
                    f_out.write(f_in.read())

        return temp_path


def get_storage() -> StorageContext:
    """Factory function to get storage context based on mode."""
    mode = settings.MODE
    if mode not in ["local", "cloud"]:
        raise ValueError(f"Invalid MODE: {mode}. Must be 'local' or 'cloud'")

    logger.info("Storage context initialized", mode=mode)
    return StorageContext(mode=mode)
