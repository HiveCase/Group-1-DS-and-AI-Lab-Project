from __future__ import annotations

from typing import Any


class ReportConsistencyService:
    """Audits the report-synthesis LLM's own output against the evidence it
    was given, instead of trusting its self-reported `recommendation_reason`
    at face value. This never changes `recommendation` -- it is a read-only
    verification pass, purely for observability (traced to Langfuse by the
    orchestrator), that catches cases where the LLM's stated reasoning
    doesn't actually follow from `policy_findings`/`fraud_assessment`, or
    where a citation doesn't exist in the evidence it was grounded in.
    """

    def check(
        self,
        report_json: dict[str, Any] | None,
        policy_findings: list[dict[str, Any]] | None,
        fraud_assessment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        report_json = report_json or {}
        policy_findings = policy_findings or []
        fraud_assessment = fraud_assessment or {}

        checks = [
            self._check_citations_grounded(report_json, policy_findings),
            self._check_sub_limit_respected(report_json, policy_findings),
            self._check_coverage_basis_present(report_json, policy_findings),
            self._check_fraud_signals_reflected(report_json, fraud_assessment),
        ]
        return {
            "checks": checks,
            "all_passed": all(check["passed"] for check in checks),
        }

    def _check_citations_grounded(self, report_json: dict[str, Any], policy_findings: list[dict[str, Any]]) -> dict[str, Any]:
        known_clause_ids = {finding.get("clause_id") for finding in policy_findings if finding.get("clause_id")}
        cited_ids: set[str] = set()
        for coverage in report_json.get("applicable_coverage") or []:
            for citation in coverage.get("citations") or []:
                clause_id = citation.get("clause_id")
                if clause_id:
                    cited_ids.add(clause_id)

        ungrounded = sorted(cited_ids - known_clause_ids)
        return {
            "rule": "citations_grounded",
            "passed": not ungrounded,
            "detail": (
                "Every cited clause_id exists in the retrieved policy_findings."
                if not ungrounded
                else f"Cited clause_id(s) not present in policy_findings (likely invented): {ungrounded}"
            ),
        }

    def _check_sub_limit_respected(self, report_json: dict[str, Any], policy_findings: list[dict[str, Any]]) -> dict[str, Any]:
        outside_limit = any(
            finding.get("clause_type") == "sub_limit" and finding.get("status") == "outside_policy_limit"
            for finding in policy_findings
        )
        recommendation = report_json.get("recommendation")
        violated = outside_limit and recommendation == "Approve"
        return {
            "rule": "sub_limit_respected",
            "passed": not violated,
            "detail": (
                "Claimed amount is outside the policy limit but recommendation is Approve."
                if violated
                else "No policy-limit violation, or recommendation already reflects it."
            ),
        }

    def _check_coverage_basis_present(self, report_json: dict[str, Any], policy_findings: list[dict[str, Any]]) -> dict[str, Any]:
        has_coverage_clause = any(finding.get("clause_type") in ("coverage", "definition") for finding in policy_findings)
        recommendation = report_json.get("recommendation")
        violated = recommendation == "Approve" and not has_coverage_clause
        return {
            "rule": "coverage_basis_present",
            "passed": not violated,
            "detail": (
                "Recommendation is Approve but no retrieved clause actually grants coverage."
                if violated
                else "A coverage clause supports the recommendation, or recommendation isn't Approve."
            ),
        }

    def _check_fraud_signals_reflected(self, report_json: dict[str, Any], fraud_assessment: dict[str, Any]) -> dict[str, Any]:
        fraud_flagged = bool(fraud_assessment.get("needs_investigation"))
        recommendation = report_json.get("recommendation")
        already_overridden = "original_recommendation" in report_json
        violated = fraud_flagged and recommendation == "Approve" and not already_overridden
        return {
            "rule": "fraud_signals_reflected",
            "passed": not violated,
            "detail": (
                "Fraud assessment flagged this claim for investigation, but the recommendation "
                "is still Approve and was never overridden."
                if violated
                else "Recommendation is consistent with the fraud assessment."
            ),
        }
