from __future__ import annotations

from typing import Any

from app.services.damage_detection_service import DamageDetectionService
from app.services.fraud_agent_service import FraudAgentService
from app.services.langgraph_orchestrator import LangGraphClaimOrchestrator
from app.services.policy_clause_service import PolicyClauseService
from app.services.report_synthesis_service import ReportSynthesisService
from app.services.severity_scoring_service import SeverityScoringService


class ClaimAnalysisOrchestrator:
    def __init__(self, damage_service: DamageDetectionService | None = None, severity_service: SeverityScoringService | None = None, policy_service: PolicyClauseService | None = None, report_service: ReportSynthesisService | None = None, fraud_service: FraudAgentService | None = None):
        self.damage_service = damage_service or DamageDetectionService()
        self.severity_service = severity_service or SeverityScoringService()
        self.policy_service = policy_service or PolicyClauseService()
        self.report_service = report_service or ReportSynthesisService()
        self.fraud_service = fraud_service or FraudAgentService()
        self.graph = LangGraphClaimOrchestrator(
            damage_service=self.damage_service,
            severity_service=self.severity_service,
            policy_service=self.policy_service,
            report_service=self.report_service,
            fraud_service=self.fraud_service,
        ).graph
        self._engine = LangGraphClaimOrchestrator(
            damage_service=self.damage_service,
            severity_service=self.severity_service,
            policy_service=self.policy_service,
            report_service=self.report_service,
            fraud_service=self.fraud_service,
        )

    def run(self, claim: Any, policy: Any = None) -> dict[str, Any]:
        return self._engine.run(claim, policy)
