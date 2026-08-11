from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.services.damage_detection_service import DamageDetectionService
from app.services.fraud_agent_service import FraudAgentService
from app.services.policy_clause_service import PolicyClauseService
from app.services.report_synthesis_service import ReportSynthesisService
from app.services.severity_scoring_service import SeverityScoringService


def build_toolkit(damage_service: DamageDetectionService | None = None, severity_service: SeverityScoringService | None = None, policy_service: PolicyClauseService | None = None, report_service: ReportSynthesisService | None = None, fraud_service: FraudAgentService | None = None) -> list[Any]:
    damage_service = damage_service or DamageDetectionService()
    severity_service = severity_service or SeverityScoringService()
    policy_service = policy_service or PolicyClauseService()
    report_service = report_service or ReportSynthesisService()
    fraud_service = fraud_service or FraudAgentService()

    @tool
    def detect_damage_tool(image_path: str) -> list[dict[str, Any]]:
        """Detect damage classes, bounding boxes, and confidence scores from an image path."""
        return damage_service.detect_from_path(image_path)

    @tool
    def score_severity_tool(detections: list[dict[str, Any]], image_width: int = 100, image_height: int = 100) -> dict[str, Any]:
        """Score the severity of a set of detections using area-ratio heuristics."""
        return severity_service.score_detections(detections, image_width=image_width, image_height=image_height)

    @tool
    def retrieve_policy_clauses_tool(
        policy_number: str | None,
        damage_classes: list[str],
        incident_description: str,
        claimed_amount: str | None,
        policy_limit: str | None,
    ) -> list[dict[str, Any]]:
        """Retrieve source-cited policy clauses for the claim's detected damage classes and check the claimed amount against the policy limit."""
        return policy_service.retrieve_clauses(policy_number, damage_classes, incident_description, claimed_amount, policy_limit)

    @tool
    def synthesize_report_tool(detections: list[dict[str, Any]], severity_summary: dict[str, Any], policy_findings: list[dict[str, Any]]) -> dict[str, Any]:
        """Synthesize a structured claim report from detections, severity, and policy findings."""
        return report_service.synthesize_report(detections, severity_summary, policy_findings)

    @tool
    def assess_fraud_tool(claim_type: str, claimed_amount: str | None, severity_summary: dict[str, Any], confidence_score: float) -> dict[str, Any]:
        """Assess fraud risk for a claim based on severity, amount, and report confidence."""
        class _ClaimProxy:
            def __init__(self, claim_type: str, claimed_amount: str | None) -> None:
                self.incident_description = claim_type
                self.claimed_amount = claimed_amount
        claim_proxy = _ClaimProxy(claim_type, claimed_amount)
        return fraud_service.assess_fraud_risk(claim_proxy, severity_summary, {"confidence_score": confidence_score})

    return [
        detect_damage_tool,
        score_severity_tool,
        retrieve_policy_clauses_tool,
        synthesize_report_tool,
        assess_fraud_tool,
    ]
