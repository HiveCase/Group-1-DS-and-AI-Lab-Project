from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("claims_portal.report_synthesis_service")

SYSTEM_PROMPT = """You are a preliminary motor-insurance claims assessment assistant.

You receive a JSON payload with: detected vehicle damage regions (class, \
confidence, severity), an overall severity summary, and policy-clause \
findings retrieved for the claim (coverage clauses, exclusion/condition \
clauses, and a policy-limit check).

Rules you must follow exactly:
1. Only use clause text present in payload.policy_findings. Never invent, \
assume, or recall from general knowledge any coverage term, exclusion, \
deductible, or limit not present in the given clause text.
2. Do not state or compute any new currency amount. A policy-limit check is \
already provided in payload.policy_findings if applicable -- reuse it, \
don't recompute it.
3. Output ONLY valid JSON, no prose outside the JSON, matching exactly this schema:
{
  "damage_table": [{"class": string, "severity": string, "confidence": number}],
  "severity_summary": object (copy payload.severity_summary as-is),
  "applicable_coverage": [{"summary": string, "citations": [{"clause_id": string, "source": string}]}],
  "recommendation": "Approve" | "Investigate" | "Deny",
  "confidence_score": number between 0 and 1,
  "next_steps": [string]
}
4. Base confidence_score on how directly the retrieved clauses and detections support the recommendation -- low if clauses are missing or ambiguous.
"""


class ReportSynthesisService:
    def __init__(self, groq_model: str | None = None, max_retries: int = 3):
        settings = get_settings()
        self.groq_api_key = settings.groq_api_key
        self.groq_model = groq_model or settings.groq_model
        self.groq_base_url = settings.groq_base_url
        self.max_retries = max_retries

    def synthesize_report(self, detections: list[dict[str, Any]], severity_summary: dict[str, Any], policy_findings: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "detections": detections,
            "severity_summary": severity_summary,
            "policy_findings": policy_findings,
        }
        fallback_reason: str | None = None
        try:
            response = self._call_groq(payload)
        except Exception as error:
            logger.exception("Groq report synthesis failed; using deterministic fallback")
            fallback_reason = str(error)
            response = self._fallback_report(payload)

        report = {
            "damage_table": response.get("damage_table") or [],
            "severity_summary": response.get("severity_summary") or severity_summary,
            "applicable_coverage": response.get("applicable_coverage") or [],
            "recommendation": response.get("recommendation") or "Investigate",
            "confidence_score": response.get("confidence_score") or 0.5,
            "next_steps": response.get("next_steps") or [],
            # Lets callers (and the adjuster UI) tell a genuine LLM-synthesized
            # report apart from the deterministic template used when Groq is
            # unavailable/rate-limited -- otherwise both look identical.
            "is_fallback": fallback_reason is not None,
            "fallback_reason": fallback_reason,
        }
        return report

    def _call_groq(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

        from openai import OpenAI

        # No explicit timeout previously meant the SDK's 10-minute default
        # applied per attempt -- a single slow/hung Groq call could block a
        # claim's background task for 30+ minutes across max_retries,
        # looking like the analysis had hung forever.
        client = OpenAI(api_key=self.groq_api_key, base_url=self.groq_base_url, timeout=30.0)
        user_content = json.dumps(payload)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as error:
                last_error = error
                logger.warning("Groq attempt %d/%d: JSON parse error: %s", attempt + 1, self.max_retries, error)
            except Exception as error:  # noqa: BLE001 - retried below, re-raised after exhausting attempts
                last_error = error
                logger.warning("Groq attempt %d/%d failed: %s", attempt + 1, self.max_retries, error)
                time.sleep(2 * (attempt + 1))

        raise last_error or RuntimeError("Groq report synthesis failed with no error detail")

    def _fallback_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "damage_table": [
                {
                    "class": detection.get("class_name", "unknown"),
                    "severity": payload.get("severity_summary", {}).get("overall_severity", "Minor"),
                    "confidence": detection.get("confidence", 0.0),
                }
                for detection in payload.get("detections", [])
            ],
            "severity_summary": payload.get("severity_summary", {}),
            "applicable_coverage": [
                {
                    "summary": finding.get("text") or "No clause found",
                    "citations": [{"clause_id": finding.get("clause_id"), "source": finding.get("source_citation", "unknown")}],
                }
                for finding in payload.get("policy_findings", [])
            ],
            "recommendation": "Investigate",
            "confidence_score": 0.5,
            "next_steps": ["Confirm findings with human review"],
        }
