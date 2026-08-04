from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(default="sqlite:///./backend/data/claims.db")
    upload_dir: Path = Field(default=Path("uploads"))
    model_dir: Path = Field(default=Path("backend/models"))
    data_dir: Path = Field(default=Path("backend/data"))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
