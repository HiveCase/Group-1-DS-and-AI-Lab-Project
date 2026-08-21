from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("claims_portal.narrative_risk_analyzer_service")

SYSTEM_PROMPT = """You are a narrative red-flag extractor for motor-insurance claims.

You receive a claim's free-text incident description. Identify short \
phrases that suggest fraud risk: staged-incident language, admission of \
pre-existing or prior damage, pressure/urgency for a fast settlement, an \
inconsistent or vague sequence of events, or explicit mention of arson/fire/ \
fraud. Do not infer meaning beyond what is literally written -- do not guess \
at intent, and do not flag ordinary, unremarkable descriptions of an \
accident just because damage occurred.

Rules you must follow exactly:
1. Each "quote" you output MUST be an exact, verbatim substring copied from \
the incident description -- never paraphrase, translate, summarize, or \
alter it in any way (not even punctuation or case).
2. If nothing in the text qualifies, return an empty list. Do not invent a \
flag to have something to report.
3. Output ONLY valid JSON, no prose outside the JSON, matching exactly this \
schema:
{"flags": [{"quote": string, "concern": string}]}
"""


class NarrativeRiskAnalyzerService:
    """Extracts fraud-relevant red-flag phrases from a claim's free-text
    incident_description -- the one part of a claim the deterministic
    FraudAgentService can only check with two hardcoded keywords ("fire",
    "fraud"). Every returned quote is verified as a verbatim substring of
    the original text before being trusted: an LLM claiming to quote text
    that isn't actually there would otherwise be an ungrounded, unverifiable
    basis for raising someone's fraud score, which is a materially worse
    failure mode than under-detecting risk in free text.
    """

    def __init__(self, groq_model: str | None = None, max_retries: int = 2):
        settings = get_settings()
        self.groq_api_key = settings.groq_api_key
        self.groq_model = groq_model or settings.groq_model
        self.groq_base_url = settings.groq_base_url
        self.max_retries = max_retries

    def analyze(self, incident_description: str) -> dict[str, Any]:
        text = (incident_description or "").strip()
        if not text:
            return {"flags": [], "available": True, "reason": None}
        if not self.groq_api_key:
            return {"flags": [], "available": False, "reason": "GROQ_API_KEY is not configured"}

        try:
            raw_flags = self._call_groq(text)
        except Exception as error:
            logger.warning("Narrative risk analysis failed; treating as no signal: %s", error)
            return {"flags": [], "available": False, "reason": str(error)}

        grounded = self._ground_flags(raw_flags, text)
        return {"flags": grounded, "available": True, "reason": None}

    def _ground_flags(self, raw_flags: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
        lowered_text = text.lower()
        grounded: list[dict[str, Any]] = []
        for flag in raw_flags:
            if not isinstance(flag, dict):
                continue
            quote = str(flag.get("quote") or "").strip()
            if quote and quote.lower() in lowered_text:
                grounded.append({"quote": quote, "concern": str(flag.get("concern") or "unspecified")})
        return grounded

    def _call_groq(self, text: str) -> list[dict[str, Any]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.groq_api_key, base_url=self.groq_base_url, timeout=15.0)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)
                return data.get("flags") or []
            except Exception as error:  # noqa: BLE001 - retried, re-raised after exhausting attempts
                last_error = error
                logger.warning("Narrative risk analysis attempt %d/%d failed: %s", attempt + 1, self.max_retries, error)

        raise last_error or RuntimeError("Narrative risk analysis failed with no error detail")
