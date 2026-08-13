from functools import lru_cache
from pathlib import Path
import os
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]  # points to backend/
DEFAULT_DB_PATH = (BASE_DIR / "data" / "claims.db").resolve()


class Settings(BaseSettings):
    database_url: str = Field(default=f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

    upload_dir: Path = Field(default=BASE_DIR / "data" / "uploads")
    model_dir: Path = Field(default=BASE_DIR / "models")
    data_dir: Path = Field(default=BASE_DIR / "data")

    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY") or None
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY") or None
    # .env historically used LANGFUSE_BASE_URL; the field is langfuse_host to
    # match the Langfuse SDK's own `host` kwarg, so both names are accepted.
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )

    groq_api_key: str | None = None
    groq_model: str = os.getenv("MODEL_NAME") or "llama-3.3-70b-versatile"
    groq_base_url: str = os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"

    jwt_secret_key: str = "change-me-in-production"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
