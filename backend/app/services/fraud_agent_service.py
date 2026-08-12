from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any


class FraudAgentService:
    def __init__(self):
        self.threshold = 0.65

    def assess_fraud_risk(
        self,
        claimant_name: str | None = None,
        claimed_amount: Any = None,
        incident_description: str | None = None,
        severity_summary: dict[str, Any] | None = None,
        confidence_score: float | None = None,
        policy_holder_name: str | None = None,
        policy_status: str | None = None,
        incident_date: Any = None,
        policy_expiry_date: Any = None,
        already_claimed_amount: Any = None,
        policy_limit: Any = None,
    ) -> dict[str, Any]:
        claim_amount = self._to_decimal(claimed_amount)
        confidence = confidence_score if confidence_score is not None else 0.5
        severity_score = (severity_summary or {}).get("severity_score", 0.0) or 0.0
        description = incident_description or ""

        fraud_score = min(0.99, max(0.15, float(severity_score) + 0.1 + (0.01 if claim_amount > 10000 else 0.0)))
        # Hard rule violations against policy data force investigation
        # regardless of the numeric score -- they are facts, not heuristics.
        signals: list[str] = []

        if "fire" in description.lower() or "fraud" in description.lower():
            fraud_score = min(0.99, fraud_score + 0.15)

        name_mismatch = self._names_mismatch(claimant_name, policy_holder_name)
        if name_mismatch:
            fraud_score = min(0.99, fraud_score + 0.25)
            signals.append(f"Claimant name '{claimant_name}' does not match policyholder of record '{policy_holder_name}'")

        policy_inactive = self._policy_inactive_at_incident(policy_status, incident_date, policy_expiry_date)
        if policy_inactive:
            fraud_score = min(0.99, fraud_score + 0.3)
            signals.append("Policy was expired or not active on the incident date")

        policy_limit_decimal = self._to_decimal(policy_limit)
        already_claimed = self._to_decimal(already_claimed_amount)
        exceeds_policy_limit = bool(policy_limit_decimal and (already_claimed + claim_amount) > policy_limit_decimal)
        if exceeds_policy_limit:
            signals.append(
                f"Already-claimed amount ({already_claimed}) plus this claim ({claim_amount}) "
                f"exceeds the policy limit ({policy_limit_decimal})"
            )
            fraud_score = min(0.99, fraud_score + 0.2)

        fraud_score = min(0.99, max(0.0, fraud_score * (1.0 - confidence * 0.2)))
        needs_investigation = bool(signals) or fraud_score >= self.threshold
        reason = "; ".join(signals) if signals else (
            "High severity and claim amount suggest deeper review"
            if fraud_score >= self.threshold
            else "No immediate fraud concern"
        )
        return {
            "fraud_score": round(fraud_score, 2),
            "needs_investigation": needs_investigation,
            "reason": reason,
            "signals": signals,
            # Structured flags so downstream logic (e.g. overriding a
            # recommendation the report-synthesis model made without this
            # context) can check a specific rule without fragile-matching
            # against the free-text signal strings above.
            "rule_flags": {
                "name_mismatch": name_mismatch,
                "policy_inactive": policy_inactive,
                "exceeds_policy_limit": exceeds_policy_limit,
            },
        }

    def _names_mismatch(self, claimant_name: str | None, policy_holder_name: str | None) -> bool:
        if not claimant_name or not policy_holder_name:
            return False
        claimant_tokens = set(claimant_name.strip().lower().split())
        holder_tokens = set(policy_holder_name.strip().lower().split())
        if not claimant_tokens or not holder_tokens:
            return False
        # Any shared token (e.g. a shared first or last name) counts as a
        # match, to tolerate name-order/formatting differences.
        return claimant_tokens.isdisjoint(holder_tokens)

    def _policy_inactive_at_incident(self, policy_status: str | None, incident_date: Any, policy_expiry_date: Any) -> bool:
        if policy_status and str(policy_status).lower() != "active":
            return True
        incident = self._to_date(incident_date)
        expiry = self._to_date(policy_expiry_date)
        if incident and expiry and incident > expiry:
            return True
        return False

    def _to_date(self, value: Any) -> date | None:
        if value is None or value == "" or value == "None":
            return None
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None

    def _to_decimal(self, value: Any) -> Decimal:
        if value is None or value == "" or value == "None":
            return Decimal("0")
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return Decimal("0")
