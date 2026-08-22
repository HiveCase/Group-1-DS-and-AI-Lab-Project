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
        narrative_flags: list[dict[str, Any]] | None = None,
        vehicle_no: str | None = None,
        policy_vehicle_registration_no: str | None = None,
    ) -> dict[str, Any]:
        claim_amount = self._to_decimal(claimed_amount)
        confidence = confidence_score if confidence_score is not None else 0.5
        severity_score = (severity_summary or {}).get("severity_score", 0.0) or 0.0
        description = incident_description or ""

        # score_breakdown records the running fraud_score after every step
        # below, in the exact order applied -- an explainability trail, not
        # a parallel computation. It is derived from the same fraud_score
        # variable the rest of this method already mutates, so it can never
        # drift out of sync with the number actually returned.
        breakdown: list[dict[str, Any]] = []

        def record(label: str, detail: str, triggered: bool, score_after: float) -> None:
            previous_total = breakdown[-1]["running_total"] if breakdown else 0.0
            breakdown.append({
                "label": label,
                "detail": detail,
                "triggered": triggered,
                "delta": round(score_after - previous_total, 4),
                "running_total": round(score_after, 4),
            })

        high_amount = claim_amount > 10000
        fraud_score = min(0.99, max(0.15, float(severity_score) + 0.1 + (0.01 if high_amount else 0.0)))
        base_detail = f"Detected-damage severity score ({float(severity_score):.2f}) plus a fixed baseline risk of 0.10"
        if high_amount:
            base_detail += f", plus 0.01 for a claimed amount ({claim_amount}) over $10,000"
        base_detail += "; floored at 0.15 and capped at 0.99"
        record("Base risk from severity and claim amount", base_detail, True, fraud_score)

        # Hard rule violations against policy data force investigation
        # regardless of the numeric score -- they are facts, not heuristics.
        signals: list[str] = []

        keyword_hit = "fire" in description.lower() or "fraud" in description.lower()
        if keyword_hit:
            fraud_score = min(0.99, fraud_score + 0.15)
        record(
            "Risk keyword in incident description",
            "Incident description contains 'fire' or 'fraud'",
            keyword_hit,
            fraud_score,
        )

        # LLM-extracted, verbatim-grounded red-flag phrases from the free
        # text (see NarrativeRiskAnalyzerService) -- catches what the two
        # hardcoded keywords above miss, without ever trusting an
        # ungrounded LLM claim: the caller is responsible for having
        # already verified each quote is a real substring of the
        # description before passing it in here. Capped well below any
        # single hard-rule weight so a narrative signal alone can nudge but
        # never dominate the score the way a factual policy violation does.
        narrative_flags = narrative_flags or []
        narrative_hit = bool(narrative_flags)
        if narrative_hit:
            fraud_score = min(0.99, fraud_score + min(0.15, 0.05 * len(narrative_flags)))
        record(
            "LLM-extracted narrative red flags",
            (
                f"{len(narrative_flags)} grounded red-flag phrase(s) found in the incident narrative: "
                + "; ".join(f"\"{flag.get('quote')}\" ({flag.get('concern')})" for flag in narrative_flags)
                if narrative_hit
                else "No grounded red-flag phrases found (or narrative analysis unavailable)"
            ),
            narrative_hit,
            fraud_score,
        )

        name_mismatch = self._names_mismatch(claimant_name, policy_holder_name)
        if name_mismatch:
            fraud_score = min(0.99, fraud_score + 0.25)
            signals.append(f"Claimant name '{claimant_name}' does not match policyholder of record '{policy_holder_name}'")
        record(
            "Claimant/policyholder name mismatch",
            f"Claimant '{claimant_name}' vs policyholder of record '{policy_holder_name}'",
            name_mismatch,
            fraud_score,
        )

        policy_inactive = self._policy_inactive_at_incident(policy_status, incident_date, policy_expiry_date)
        if policy_inactive:
            fraud_score = min(0.99, fraud_score + 0.3)
            signals.append("Policy was expired or not active on the incident date")
        record(
            "Policy inactive or expired at incident date",
            f"Policy status '{policy_status}', incident date {incident_date}, expiry {policy_expiry_date}",
            policy_inactive,
            fraud_score,
        )

        vehicle_mismatch = self._vehicle_mismatch(vehicle_no, policy_vehicle_registration_no)
        if vehicle_mismatch:
            fraud_score = min(0.99, fraud_score + 0.25)
            signals.append(
                f"Claimed vehicle number '{vehicle_no}' does not match the registration on the policy "
                f"of record '{policy_vehicle_registration_no}'"
            )
        record(
            "Claim/policy vehicle registration mismatch",
            f"Claim vehicle no. '{vehicle_no}' vs policy registration on record '{policy_vehicle_registration_no}'",
            vehicle_mismatch,
            fraud_score,
        )

        policy_limit_decimal = self._to_decimal(policy_limit)
        already_claimed = self._to_decimal(already_claimed_amount)
        exceeds_policy_limit = bool(policy_limit_decimal and (already_claimed + claim_amount) > policy_limit_decimal)
        if exceeds_policy_limit:
            signals.append(
                f"Already-claimed amount ({already_claimed}) plus this claim ({claim_amount}) "
                f"exceeds the policy limit ({policy_limit_decimal})"
            )
            fraud_score = min(0.99, fraud_score + 0.2)
        record(
            "Cumulative claimed amount exceeds policy limit",
            f"Already claimed {already_claimed} + this claim {claim_amount} vs. limit {policy_limit_decimal}",
            exceeds_policy_limit,
            fraud_score,
        )

        fraud_score = min(0.99, max(0.0, fraud_score * (1.0 - confidence * 0.2)))
        record(
            "Confidence dampening",
            f"Running score scaled by (1 - confidence x 0.2), where confidence={confidence:.2f} "
            "-- a more confident report synthesis slightly reduces the fraud score",
            True,
            fraud_score,
        )

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
                "vehicle_mismatch": vehicle_mismatch,
                "exceeds_policy_limit": exceeds_policy_limit,
            },
            # Explainability: the exact, ordered sequence of adjustments that
            # produced fraud_score, so an adjuster/SIU reviewer sees *how
            # much* each factor moved the score, not just that it fired.
            "score_breakdown": breakdown,
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

    def _vehicle_mismatch(self, vehicle_no: str | None, policy_vehicle_registration_no: str | None) -> bool:
        if not vehicle_no or not policy_vehicle_registration_no:
            return False
        normalize = lambda value: "".join(value.split()).replace("-", "").upper()
        return normalize(vehicle_no) != normalize(policy_vehicle_registration_no)

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
