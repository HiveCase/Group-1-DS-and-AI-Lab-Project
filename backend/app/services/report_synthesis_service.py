from __future__ import annotations

import json
from typing import Any

from app.core.config import get_settings
from app.services.langfuse_observability import LangfuseObserver


class ReportSynthesisService:
    def __init__(self, ollama_model: str | None = None):
        settings = get_settings()
        self.ollama_model = ollama_model or settings.ollama_model
        self.ollama_base_url = settings.ollama_base_url
        self.observer = LangfuseObserver()

    def synthesize_report(self, detections: list[dict[str, Any]], severity_summary: dict[str, Any], policy_findings: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "detections": detections,
            "severity_summary": severity_summary,
            "policy_findings": policy_findings,
        }
        try:
            response = self._call_ollama(payload)
        except Exception:
            response = self._fallback_report(payload)

        report = {
            "damage_table": response.get("damage_table") or [],
            "severity_summary": response.get("severity_summary") or severity_summary,
            "applicable_coverage": response.get("applicable_coverage") or [],
            "recommendation": response.get("recommendation") or "Investigate",
            "confidence_score": response.get("confidence_score") or 0.5,
            "next_steps": response.get("next_steps") or [],
        }
        self.observer.trace_generation(
            name="claim_report_synthesis",
            input_data={"detections": detections, "severity_summary": severity_summary, "policy_findings": policy_findings},
            output_data=report,
            metadata={"model": self.ollama_model},
        )
        return report

    def _call_ollama(self, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        response = requests.post(
            self.ollama_base_url + "/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": json.dumps(payload),
                "stream": False,
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return json.loads(data.get("response", "{}"))

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
