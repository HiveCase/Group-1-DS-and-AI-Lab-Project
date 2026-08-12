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
clauses, and a policy-limit check). You do not have access to the claim \
photos or any information beyond this payload -- do not imply visual \
judgment you don't have, and do not draw on general insurance knowledge \
beyond what payload.policy_findings actually says.

Rules you must follow exactly:
1. Only use clause text present in payload.policy_findings. Never invent, \
assume, or recall from general knowledge any coverage term, exclusion, \
deductible, or limit not present in the given clause text.
2. Do not state or compute any new currency amount. A policy-limit check is \
already provided in payload.policy_findings (clause_type "sub_limit") if \
applicable -- reuse its "status" field, don't recompute it.
3. Every citations[].clause_id you output MUST be copied verbatim from a \
clause_id already present in payload.policy_findings. Never invent a \
clause_id or cite one that isn't in the payload.
4. Decide "recommendation" using this priority order, checking each in turn:
   a. If any policy_findings entry has clause_type "sub_limit" and status \
   "outside_policy_limit", recommendation must be "Deny" or "Investigate" \
   -- never "Approve".
   b. If any retrieved exclusion/condition clause plainly and directly \
   applies to a detected damage class, recommendation must be "Investigate" \
   or "Deny", not "Approve".
   c. If policy_findings is empty, or has no clause_type "coverage" entry \
   relevant to the detected damage classes, recommendation must be \
   "Investigate" (there is no coverage basis to Approve).
   d. Otherwise, if coverage clauses directly and unambiguously support the \
   detected damage with no conflicting exclusion, recommendation may be \
   "Approve".
5. For damage_table[].severity, copy the matching class's severity from \
payload.severity_summary.per_region -- do not judge severity yourself.
6. confidence_score reflects how directly the retrieved clauses and \
detections support the chosen recommendation -- low (below 0.6) if clauses \
are missing, ambiguous, or conflicting; high only when the match is direct \
and unambiguous.
7. next_steps must be specific to this claim's actual gaps or conditions \
(e.g. a specific document to verify, a specific clause condition to \
confirm) -- not generic filler like "review the claim".
8. Output ONLY valid JSON, no prose outside the JSON, matching exactly this schema:
{
  "damage_table": [{"class": string, "severity": string, "confidence": number}],
  "severity_summary": object (copy payload.severity_summary as-is),
  "applicable_coverage": [{"summary": string, "citations": [{"clause_id": string, "source": string}]}],
  "recommendation": "Approve" | "Investigate" | "Deny",
  "confidence_score": number between 0 and 1,
  "next_steps": [string]
}
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
