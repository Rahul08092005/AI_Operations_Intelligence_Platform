"""Application settings, loaded from environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> parents[2] is the repository root
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Operations Intelligence Platform"
    app_version: str = "0.1.0"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://aiops:change_me_locally@localhost:5432/aiops"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_dir: str = "models"
    data_dir: str = "data"

    llm_provider: str = ""
    llm_api_key: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
