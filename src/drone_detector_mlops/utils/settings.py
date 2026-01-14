from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # General settings
    RANDOM_SEED: int = Field(default=42)
    IMAGENET_MEAN: list = Field(default=[0.485, 0.456, 0.406])
    IMAGENET_STD: list = Field(default=[0.229, 0.224, 0.225])

    # W&B settings
    WANDB_PROJECT_NAME: str = Field("drone-detector-mlops")
    WANDB_API_KEY: Optional[str] = Field(default=None)  # Optional for API deployment

    # Cloud settings
    MODE: str = Field(default="cloud")  # "local" or "cloud"
    GCS_DATA_PATH: str = Field(default="gs://drone-detection-mlops-data/structured")
    GCS_MODELS_BUCKET: str = Field(default="gs://drone-detection-mlops-models")
    GCP_PROJECT: str = Field(default="drone-detection-mlops")
    GCP_REGION: str = Field(default="europe-west4")

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_UPLOAD_SIZE_MB: int = 10
    MODEL_FILENAME: str = "model-latest.onnx"
    API_CORS_ORIGINS: list[str] = Field(default=["*"], description="Allowed CORS origins for API requests")


settings = Settings()
