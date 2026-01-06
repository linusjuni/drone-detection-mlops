from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    RANDOM_SEED: int = Field(42, env="RANDOM_SEED")

    class Config:
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
