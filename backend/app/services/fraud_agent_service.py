from __future__ import annotations

from typing import Any


class FraudAgentService:
    def __init__(self):
        self.threshold = 0.65

    def assess_fraud_risk(self, claim: Any, severity_summary: dict[str, Any] | None = None, report_json: dict[str, Any] | None = None) -> dict[str, Any]:
        claim_amount = getattr(claim, "claimed_amount", 0) or 0
        description = getattr(claim, "incident_description", "") or ""
        severity_score = (severity_summary or {}).get("severity_score", 0.0) or 0.0
        confidence = (report_json or {}).get("confidence_score", 0.5) or 0.5

        fraud_score = min(0.99, max(0.15, float(severity_score) + 0.1 + (0.01 if claim_amount > 10000 else 0.0)))
        if "fire" in description.lower() or "fraud" in description.lower():
            fraud_score = min(0.99, fraud_score + 0.15)

        fraud_score = min(0.99, max(0.0, fraud_score * (1.0 - confidence * 0.2)))
        return {
            "fraud_score": round(fraud_score, 2),
            "needs_investigation": fraud_score >= self.threshold,
            "reason": "High severity and claim amount suggest deeper review" if fraud_score >= self.threshold else "No immediate fraud concern",
        }
