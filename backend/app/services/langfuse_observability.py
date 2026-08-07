from __future__ import annotations

from typing import Any

from app.core.config import get_settings


class LangfuseObserver:
    def __init__(self):
        settings = get_settings()
        self.enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self.client = None
        if self.enabled:
            try:
                from langfuse import Langfuse
            except Exception:
                self.enabled = False
                self.client = None
            else:
                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )

    def trace_generation(self, *, name: str, input_data: dict[str, Any], output_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.trace(
                name=name,
                input=input_data,
                output=output_data,
                metadata=metadata or {},
            )
        except Exception:
            return
