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

    RANDOM_SEED: int = Field(default=42)
    IMAGENET_MEAN: list = Field(default=[0.485, 0.456, 0.406])
    IMAGENET_STD: list = Field(default=[0.229, 0.224, 0.225])

    WANDB_PROJECT_NAME: str = Field("drone-detector-mlops")
    WANDB_API_KEY: str = Field(...)


settings = Settings()
