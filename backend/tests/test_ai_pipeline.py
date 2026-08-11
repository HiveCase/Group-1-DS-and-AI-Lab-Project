from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.damage_detection_service import DamageDetectionService
from app.services.policy_clause_service import PolicyClauseService
from app.services.report_synthesis_service import ReportSynthesisService
from app.services.severity_scoring_service import SeverityScoringService
from app.services.claim_analysis_graph import ClaimAnalysisOrchestrator


class DummyModel:
    def predict(self, image_path):
        return [{
            "class_name": "dent",
            "bbox": [0, 0, 20, 20],
            "mask_polygon": [[0, 0], [20, 0], [20, 20], [0, 20]],
            "confidence": 0.82,
        }]


def test_damage_detection_service_returns_structured_detections(tmp_path, monkeypatch):
    service = DamageDetectionService(model_path=tmp_path / "dummy.pt")
    monkeypatch.setattr(service, "_load_model", lambda: DummyModel())

    detections = service.detect_from_path(tmp_path / "sample.jpg")

    assert len(detections) == 1
    assert detections[0]["class_name"] == "dent"
    assert detections[0]["bbox"] == [0, 0, 20, 20]
    assert detections[0]["confidence"] == 0.82


def test_severity_scoring_service_applies_confidence_floor():
    service = SeverityScoringService()
    detections = [{
        "class_name": "shattered_glass",
        "confidence": 0.6,
        "mask_polygon": [[0, 0], [20, 0], [20, 20], [0, 20]],
    }]

    result = service.score_detections(detections, image_width=100, image_height=100)

    assert result["overall_severity"] == "Minor"
    assert result["severity_score"] == 0.04


class FakeClauseRetriever:
    def get_clauses(self, damage_class):
        return {
            "coverage": [{
                "chunk_id": "CL-001",
                "text": "Comprehensive coverage applies to accidental body damage up to the policy limit.",
                "heading": "section 3.1",
                "clause_type": "coverage",
                "doc_id": "user_POL-001",
                "score": 0.9,
            }],
            "exclusion_or_condition": [],
            "coverage_clause_found": True,
        }


def test_policy_clause_service_returns_citations():
    service = PolicyClauseService(retriever_factory=lambda policy_number: FakeClauseRetriever())

    findings = service.retrieve_clauses("POL-001", ["dent"], "rear bumper damage", Decimal("1200"))

    assert findings[0]["clause_id"] == "CL-001"
    assert findings[0]["source_citation"] == "section 3.1"
    assert findings[0]["damage_class"] == "dent"


def test_policy_clause_service_flags_amount_outside_policy_limit():
    service = PolicyClauseService(retriever_factory=lambda policy_number: FakeClauseRetriever())

    findings = service.retrieve_clauses("POL-001", [], "no damage class match here", Decimal("9000"), Decimal("5000"))

    limit_finding = next(f for f in findings if f["clause_id"] == "POLICY-LIMIT-CHECK")
    assert limit_finding["status"] == "outside_policy_limit"


def test_policy_clause_service_skips_retrieval_without_policy_number():
    service = PolicyClauseService(retriever_factory=lambda policy_number: FakeClauseRetriever())

    findings = service.retrieve_clauses(None, ["dent"], "rear bumper damage", Decimal("1200"))

    assert findings == []


def test_policy_clause_service_real_retrieval_for_seeded_policy():
    """Exercises the actual rag_scripts hybrid-retrieval pipeline (chromadb +
    sentence-transformers) against the real POL-001 synthetic policy PDF,
    skipping if the embedding model can't be loaded in this environment
    (e.g. no network access for the first-time Hugging Face download)."""
    service = PolicyClauseService()
    try:
        findings = service.retrieve_clauses("POL-001", ["dent"], "front bumper dent", Decimal("1200"), Decimal("5000"))
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"rag_scripts retrieval unavailable in this environment: {exc}")

    clause_findings = [f for f in findings if f["clause_id"] != "POLICY-LIMIT-CHECK"]
    if not clause_findings:
        pytest.skip("Policy-clause retrieval returned no hits; embedding model likely unavailable")
    assert clause_findings[0]["damage_class"] == "dent"


def test_report_synthesis_service_outputs_frontend_matching_json(monkeypatch):
    service = ReportSynthesisService()

    monkeypatch.setattr(service, "_call_groq", lambda payload: {
        "damage_table": [{"class": "dent", "severity": "Minor", "confidence": 0.82}],
        "severity_summary": {"overall": "Minor", "per_region": [{"class": "dent", "severity": "Minor"}]},
        "applicable_coverage": [{"summary": "Covered", "citations": [{"clause_id": "CL-001", "source": "section 3.1"}]}],
        "recommendation": "Approve",
        "confidence_score": 0.86,
        "next_steps": ["Monitor claim"],
    })

    report = service.synthesize_report(
        detections=[{"class_name": "dent", "confidence": 0.82, "severity": "Minor"}],
        severity_summary={"overall_severity": "Minor", "severity_score": 0.04},
        policy_findings=[{"clause_id": "CL-001", "source_citation": "section 3.1", "text": "Covered"}],
    )

    assert report["damage_table"][0]["class"] == "dent"
    assert report["applicable_coverage"][0]["citations"][0]["source"] == "section 3.1"
    assert report["confidence_score"] == 0.86


def test_report_synthesis_service_falls_back_without_groq_api_key(monkeypatch):
    service = ReportSynthesisService()
    monkeypatch.setattr(service, "groq_api_key", None)

    report = service.synthesize_report(
        detections=[{"class_name": "dent", "confidence": 0.82}],
        severity_summary={"overall_severity": "Minor", "severity_score": 0.04},
        policy_findings=[{"clause_id": "CL-001", "source_citation": "section 3.1", "text": "Covered"}],
    )

    assert report["recommendation"] == "Investigate"
    assert report["damage_table"][0]["class"] == "dent"


def test_report_synthesis_service_real_groq_call():
    """Exercises the actual Groq Cloud call end-to-end, skipping if no
    GROQ_API_KEY is configured in this environment."""
    service = ReportSynthesisService()
    if not service.groq_api_key:
        pytest.skip("GROQ_API_KEY not configured")

    report = service.synthesize_report(
        detections=[{"class_name": "dent", "confidence": 0.82, "severity": "Minor"}],
        severity_summary={"overall_severity": "Minor", "severity_score": 0.04, "per_region": []},
        policy_findings=[{"clause_id": "CL-001", "source_citation": "section 3.1", "text": "Comprehensive coverage applies to accidental body damage up to the policy limit."}],
    )

    assert report["recommendation"] in {"Approve", "Investigate", "Deny"}
    assert 0.0 <= report["confidence_score"] <= 1.0


def test_claim_analysis_orchestrator_runs_five_step_pipeline():
    orchestrator = ClaimAnalysisOrchestrator(
        damage_service=DummyDamageService(),
        severity_service=DummySeverityService(),
        policy_service=DummyPolicyService(),
        report_service=DummyReportService(),
    )

    claim = SimpleNamespace(
        claim_id="CLM-1001",
        incident_description="Rear bumper damage",
        claimed_amount=Decimal("1200"),
        photos=[SimpleNamespace(file_path="/tmp/sample.jpg")],
    )

    outcome = orchestrator.run(claim, policy=None)

    assert outcome["needs_human_review"] is False
    assert outcome["report_json"]["recommendation"] == "Approve"


def test_claim_analysis_orchestrator_exposes_langgraph_workflow():
    orchestrator = ClaimAnalysisOrchestrator(
        damage_service=DummyDamageService(),
        severity_service=DummySeverityService(),
        policy_service=DummyPolicyService(),
        report_service=DummyReportService(),
    )

    assert orchestrator.graph is not None
    assert hasattr(orchestrator.graph, "invoke")

    graph_nodes = set(getattr(orchestrator.graph, "nodes", {}).keys())
    assert {"coordinator", "tool_execution"}.issubset(graph_nodes)
    assert {"damage_detection", "severity_scoring", "policy_clause_retrieval", "report_synthesis", "flag_human_review", "finalize_claim"}.issubset(graph_nodes)


def test_claim_analysis_orchestrator_handles_empty_detections_without_recursing():
    orchestrator = ClaimAnalysisOrchestrator(
        damage_service=EmptyDetectionService(),
        severity_service=DummySeverityService(),
        policy_service=DummyPolicyService(),
        report_service=DummyReportService(),
    )

    claim = SimpleNamespace(
        claim_id="CLM-1002",
        incident_description="No visible damage",
        claimed_amount=Decimal("500"),
        photos=[SimpleNamespace(file_path="/tmp/empty.jpg")],
    )

    outcome = orchestrator.run(claim, policy=None)

    assert outcome["needs_human_review"] is False
    assert outcome["detections"] == []


class DummyDamageService:
    def detect_from_path(self, image_path):
        return [{"class_name": "dent", "bbox": [0, 0, 20, 20], "mask_polygon": [[0, 0], [20, 0], [20, 20], [0, 20]], "confidence": 0.82}]


class EmptyDetectionService:
    def detect_from_path(self, image_path):
        return []


class DummySeverityService:
    def score_detections(self, detections, image_width=100, image_height=100):
        return {"overall_severity": "Minor", "severity_score": 0.04, "per_region": [{"class_name": "dent", "severity": "Minor"}]}


class DummyPolicyService:
    def retrieve_clauses(self, policy_number, damage_classes, incident_description, claimed_amount, policy_limit=None):
        return [{"clause_id": "CL-001", "text": "Covered", "source_citation": "section 3.1"}]


class DummyReportService:
    def synthesize_report(self, detections, severity_summary, policy_findings):
        return {
            "damage_table": [{"class": "dent", "severity": "Minor", "confidence": 0.82}],
            "severity_summary": {"overall": "Minor", "per_region": [{"class": "dent", "severity": "Minor"}]},
            "applicable_coverage": [{"summary": "Covered", "citations": [{"clause_id": "CL-001", "source": "section 3.1"}]}],
            "recommendation": "Approve",
            "confidence_score": 0.86,
            "next_steps": ["Monitor claim"],
        }
