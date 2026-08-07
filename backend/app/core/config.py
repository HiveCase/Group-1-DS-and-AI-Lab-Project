from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]  # points to backend/
DEFAULT_DB_PATH = (BASE_DIR / "data" / "claims.db").resolve()


class Settings(BaseSettings):
    database_url: str = Field(default=f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

    upload_dir: Path = Field(default=BASE_DIR / "data" / "uploads")
    model_dir: Path = Field(default=BASE_DIR / "models")
    data_dir: Path = Field(default=BASE_DIR / "data")

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
