from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("claims_portal.langfuse_observability")


class LangfuseObserver:
    """Thin wrapper around the Langfuse v2 client.

    One trace is meant to represent one claim-analysis pipeline run, with
    each of the five agents (damage, severity, policy, fraud, report)
    logged as a span/generation nested under that trace -- not as five
    disconnected root-level traces, which is what a naive per-service
    `client.trace(...)` call would produce.
    """

    def __init__(self):
        settings = get_settings()
        self.enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self.client = None
        if self.enabled:
            try:
                from langfuse import Langfuse
            except Exception:
                logger.warning("langfuse package unavailable; observability disabled")
                self.enabled = False
            else:
                try:
                    # Bounded so a slow/unreachable Langfuse endpoint can't
                    # make flush() (called at the end of every claim's
                    # analysis) retry for minutes -- observability must
                    # never be able to block a claim from completing.
                    self.client = Langfuse(
                        public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key,
                        host=settings.langfuse_host,
                        timeout=10,
                        max_retries=2,
                    )
                except Exception:
                    logger.exception("Failed to initialize Langfuse client")
                    self.enabled = False
                    self.client = None

    def start_trace(self, *, name: str, input_data: Any, metadata: dict[str, Any] | None = None) -> Any | None:
        if not self.enabled or self.client is None:
            return None
        try:
            return self.client.trace(name=name, input=input_data, metadata=metadata or {})
        except Exception:
            logger.exception("Failed to start Langfuse trace %s", name)
            return None

    def span(self, parent: Any | None, *, name: str, input_data: Any, output_data: Any, metadata: dict[str, Any] | None = None) -> Any | None:
        """parent is either the trace (for a top-level agent span) or a span
        already returned by this method (to nest a child span underneath
        it, e.g. an explainability span under the agent span it explains).
        Both expose the same .span(...) method, so this doesn't need to
        know which kind it was given. Returns the created span so a caller
        can pass it back in as a parent for its own children, or None if
        logging is disabled/failed -- callers must tolerate None the same
        way they already tolerate a None trace."""
        if parent is None:
            return None
        try:
            return parent.span(name=name, input=input_data, output=output_data, metadata=metadata or {})
        except Exception:
            logger.exception("Failed to log Langfuse span %s", name)
            return None

    def generation(self, parent: Any | None, *, name: str, input_data: Any, output_data: Any, model: str | None = None, metadata: dict[str, Any] | None = None) -> Any | None:
        if parent is None:
            return None
        try:
            return parent.generation(name=name, input=input_data, output=output_data, model=model, metadata=metadata or {})
        except Exception:
            logger.exception("Failed to log Langfuse generation %s", name)
            return None

    def end_trace(self, trace: Any | None, *, output_data: Any) -> None:
        if trace is None:
            return
        try:
            trace.update(output=output_data)
        except Exception:
            logger.exception("Failed to finalize Langfuse trace")

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                logger.exception("Failed to flush Langfuse client")
