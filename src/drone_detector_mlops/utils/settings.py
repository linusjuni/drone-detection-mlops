from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    RANDOM_SEED: int = Field(42, env="RANDOM_SEED")

    IMAGENET_MEAN: list = Field([0.485, 0.456, 0.406], env="IMAGENET_MEAN")
    IMAGENET_STD: list = Field([0.229, 0.224, 0.225], env="IMAGENET_STD")

    WANDB_PROJECT_NAME: str = Field("drone-detector-mlops", env="WANDB_PROJECT_NAME")
    WANDB_API_KEY: str = Field(..., env="WANDB_API_KEY")

    class Config:
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
