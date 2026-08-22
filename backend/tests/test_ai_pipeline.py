from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.damage_detection_service import DamageDetectionService
from app.services.policy_clause_service import PolicyClauseService
from app.services.report_synthesis_service import ReportSynthesisService
from app.services.severity_scoring_service import SeverityScoringService
from app.services.claim_analysis_graph import ClaimAnalysisOrchestrator
from app.services.fraud_agent_service import FraudAgentService
from app.services.langgraph_orchestrator import LangGraphClaimOrchestrator
from app.services.narrative_risk_analyzer_service import NarrativeRiskAnalyzerService
from app.services.report_consistency_service import ReportConsistencyService


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


def test_severity_scoring_service_image_area_overrides_width_height():
    """A real photo's bbox coordinates (e.g. ~600x400px) divided by the old
    hardcoded 100x100 default produced ratios far above 1.0, saturating
    every claim's severity to 'Severe' regardless of actual damage size.
    image_area must be honored over image_width * image_height."""
    service = SeverityScoringService()
    detections = [{
        "class_name": "dent",
        "confidence": 0.9,
        "bbox": [0, 0, 200, 200],
    }]

    result = service.score_detections(detections, image_width=100, image_height=100, image_area=640 * 480)

    assert result["severity_score"] == round(40000 / (640 * 480), 4)
    assert result["overall_severity"] == "Moderate"


def test_damage_detection_service_reads_real_image_dimensions(tmp_path):
    from PIL import Image

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (640, 480), color="white").save(image_path)

    service = DamageDetectionService(model_path=tmp_path / "dummy.pt")

    assert service.get_image_dimensions(image_path) == (640, 480)


def test_damage_detection_service_get_image_dimensions_missing_file(tmp_path):
    service = DamageDetectionService(model_path=tmp_path / "dummy.pt")

    assert service.get_image_dimensions(tmp_path / "does-not-exist.jpg") == (0, 0)


class OcclusionAwareModel:
    """Stands in for YOLO in the occlusion-sensitivity test: instead of real
    CV, it looks at one fixed pixel and reports high confidence only while
    that pixel still holds its original (non-occluded) color -- letting the
    test assert explain_detection() correctly identifies which grid cell
    covers that pixel as the most important one, without a real model."""

    def __init__(self, feature_point, bbox, feature_color=(0, 0, 255)):
        self.feature_point = feature_point
        self.bbox = bbox
        self.feature_color = feature_color

    def predict(self, image, stream=None, imgsz=None):
        pixel = image.convert("RGB").getpixel(self.feature_point)
        confidence = 0.9 if pixel == self.feature_color else 0.05
        return [{
            "class_name": "dent",
            "bbox": list(self.bbox),
            "mask_polygon": None,
            "confidence": confidence,
        }]


def test_damage_detection_service_explain_detection_finds_important_cell(tmp_path, monkeypatch):
    from PIL import Image

    # PNG (lossless), not JPEG -- JPEG's block compression would blur the
    # single feature pixel into a neighboring color on round-trip, so the
    # fake model's exact-pixel-match check would misfire on cells that
    # never actually touched it.
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (64, 64), color=(200, 200, 200))
    bbox = (10, 10, 50, 50)
    feature_point = (15, 15)  # inside the bbox's top-left grid cell only
    image.putpixel(feature_point, (0, 0, 255))
    image.save(image_path)

    service = DamageDetectionService(model_path=tmp_path / "dummy.pt")
    monkeypatch.setattr(service, "_load_model", lambda: OcclusionAwareModel(feature_point, bbox))

    detection = {"class_name": "dent", "bbox": list(bbox), "confidence": 0.9}
    saliency = service.explain_detection(image_path, detection, grid_size=4)

    assert saliency is not None
    assert saliency["method"] == "occlusion_sensitivity"
    assert saliency["peak_cell"] == {"row": 0, "col": 0}
    assert saliency["importance"][0][0] == 1.0
    # every other cell's occlusion never touches the feature pixel, so the
    # model's reported confidence -- and therefore the measured drop -- is
    # unchanged for all of them.
    flattened_elsewhere = [
        value
        for row_index, row in enumerate(saliency["importance"])
        for col_index, value in enumerate(row)
        if (row_index, col_index) != (0, 0)
    ]
    assert all(value == 0.0 for value in flattened_elsewhere)
    assert saliency["peak_confidence_drop"] == pytest.approx(0.85)


def test_damage_detection_service_explain_detection_returns_none_without_model(tmp_path, monkeypatch):
    service = DamageDetectionService(model_path=tmp_path / "dummy.pt")
    monkeypatch.setattr(service, "_load_model", lambda: None)

    result = service.explain_detection(tmp_path / "sample.jpg", {"class_name": "dent", "bbox": [0, 0, 10, 10], "confidence": 0.9})

    assert result is None


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
    assert findings[0]["retrieval_breakdown"] is None


class FakeClauseRetrieverWithBreakdown:
    def get_clauses(self, damage_class):
        return {
            "coverage": [{
                "chunk_id": "CL-001",
                "text": "Comprehensive coverage applies to accidental body damage up to the policy limit.",
                "heading": "section 3.1",
                "clause_type": "coverage",
                "doc_id": "user_POL-001",
                "score": 0.9,
                "retrieval_breakdown": {
                    "dense_rank": 1, "dense_contribution": 0.0492,
                    "sparse_rank": None, "sparse_contribution": 0.0,
                },
            }],
            "exclusion_or_condition": [],
            "coverage_clause_found": True,
        }


def test_policy_clause_service_propagates_retrieval_breakdown():
    """The dense/sparse retrieval explanation from HybridRetriever must
    survive PolicyClauseService's own findings shape, not just the
    ClauseRetriever's internal hit dict, since it's the findings shape that
    reaches the adjuster UI and the report-synthesis LLM payload."""
    service = PolicyClauseService(retriever_factory=lambda policy_number: FakeClauseRetrieverWithBreakdown())

    findings = service.retrieve_clauses("POL-001", ["dent"], "rear bumper damage", Decimal("1200"))

    assert findings[0]["retrieval_breakdown"] == {
        "dense_rank": 1, "dense_contribution": 0.0492,
        "sparse_rank": None, "sparse_contribution": 0.0,
    }


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
        "recommendation_reason": "Clause CL-001 directly covers the detected dent with no conflicting exclusion.",
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
    assert report["is_fallback"] is False
    assert report["fallback_reason"] is None
    assert "CL-001" in report["recommendation_reason"]


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
    assert report["is_fallback"] is True
    assert "GROQ_API_KEY" in report["fallback_reason"]
    assert report["recommendation_reason"]


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
    assert report["recommendation_reason"], "the live model must justify its recommendation, not just state it"


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
    assert {"coordinator", "damage_detection", "severity_scoring", "policy_clause_retrieval", "report_synthesis", "fraud_assessment", "flag_human_review", "finalize_claim"}.issubset(graph_nodes)


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


def _build_engine(**overrides):
    from app.services.fraud_agent_service import FraudAgentService

    services = dict(
        damage_service=DummyDamageService(),
        severity_service=DummySeverityService(),
        policy_service=DummyPolicyService(),
        report_service=DummyReportService(),
        fraud_service=FraudAgentService(),
    )
    services.update(overrides)
    return LangGraphClaimOrchestrator(**services)


class DimensionAwareDamageService:
    """Mirrors a real photo + YOLO model: a large-in-pixels bbox against a
    real (non-100x100) photo size, to guard the image_area wiring fix."""

    def detect_from_path(self, image_path):
        return [{"class_name": "dent", "bbox": [50, 50, 250, 250], "mask_polygon": None, "confidence": 0.9}]

    def get_image_dimensions(self, image_path):
        return (640, 480)


def test_orchestrator_uses_real_image_area_for_severity_not_hardcoded_100x100():
    from app.services.fraud_agent_service import FraudAgentService

    engine = LangGraphClaimOrchestrator(
        damage_service=DimensionAwareDamageService(),
        severity_service=SeverityScoringService(),
        policy_service=DummyPolicyService(),
        report_service=DummyReportService(),
        fraud_service=FraudAgentService(),
    )

    claim = SimpleNamespace(
        claim_id="CLM-TEST",
        incident_description="dent",
        claimed_amount=Decimal("500"),
        photos=[SimpleNamespace(file_path="/tmp/anything.jpg")],
    )

    outcome = engine.run(claim, policy=None)

    # bbox area 200*200=40000 over a real 640x480=307200 image -> ratio ~0.13
    # (Moderate), not the >1.0 "Severe" ratio the old hardcoded 100x100
    # denominator produced for any real-photo-sized bbox.
    assert outcome["severity_summary"]["severity_score"] < 1.0
    assert outcome["severity_summary"]["overall_severity"] != "Severe"


def test_valid_next_actions_dependency_graph():
    engine = _build_engine()

    assert engine._valid_next_actions({}) == ["detect_damage"]

    after_detection = {"completed_actions": ["detect_damage"]}
    assert set(engine._valid_next_actions(after_detection)) == {"score_severity", "retrieve_policy"}

    after_one = {"completed_actions": ["detect_damage", "score_severity"]}
    assert engine._valid_next_actions(after_one) == ["retrieve_policy"]

    after_both = {"completed_actions": ["detect_damage", "score_severity", "retrieve_policy"]}
    assert engine._valid_next_actions(after_both) == ["synthesize_report"]

    after_report = {"completed_actions": ["detect_damage", "score_severity", "retrieve_policy", "synthesize_report"]}
    assert engine._valid_next_actions(after_report) == ["assess_fraud"]

    after_all = {"completed_actions": ["detect_damage", "score_severity", "retrieve_policy", "synthesize_report", "assess_fraud"]}
    assert engine._valid_next_actions(after_all) == []


def test_valid_next_actions_never_retriggers_on_empty_result():
    """detect_damage legitimately produces an empty detections list when a
    claim has no photos; completed_actions membership, not truthiness of
    the data, must be the gate or this would loop forever."""
    engine = _build_engine()
    state = {"completed_actions": ["detect_damage"], "detections": []}
    assert "detect_damage" not in engine._valid_next_actions(state)


def test_coordinator_falls_back_without_groq_configured():
    engine = _build_engine()
    assert engine.groq_api_key is None

    chosen = engine._plan_next_action({"trace": None}, ["score_severity", "retrieve_policy"])

    assert chosen == "score_severity"


def test_coordinator_falls_back_when_groq_call_fails(monkeypatch):
    engine = _build_engine()
    monkeypatch.setattr(engine, "groq_api_key", "fake-key-for-test")
    monkeypatch.setattr(engine, "groq_model", "does-not-matter")

    chosen = engine._plan_next_action({"trace": None}, ["score_severity", "retrieve_policy"])

    # unreachable base_url / bad key -> the OpenAI call raises -> deterministic fallback
    assert chosen == "score_severity"


def test_coordinator_plans_via_real_groq_when_choice_exists():
    """Exercises the actual Groq tool-calling call for the coordinator's
    genuine choice point, skipping if no GROQ_API_KEY is configured."""
    from app.services.report_synthesis_service import ReportSynthesisService

    engine = _build_engine(report_service=ReportSynthesisService())
    if not engine.groq_api_key:
        pytest.skip("GROQ_API_KEY not configured")

    state = {
        "trace": None,
        "completed_actions": ["detect_damage"],
        "detections": [{"class_name": "dent", "confidence": 0.82}],
    }
    chosen = engine._plan_next_action(state, ["score_severity", "retrieve_policy"])

    assert chosen in {"score_severity", "retrieve_policy"}


class DummyDamageService:
    def detect_from_path(self, image_path):
        return [{"class_name": "dent", "bbox": [0, 0, 20, 20], "mask_polygon": [[0, 0], [20, 0], [20, 20], [0, 20]], "confidence": 0.82}]


class EmptyDetectionService:
    def detect_from_path(self, image_path):
        return []


class DummySeverityService:
    def score_detections(self, detections, image_width=100, image_height=100, image_area=None):
        return {"overall_severity": "Minor", "severity_score": 0.04, "per_region": [{"class_name": "dent", "severity": "Minor"}]}


class DummyPolicyService:
    def retrieve_clauses(self, policy_number, damage_classes, incident_description, claimed_amount, policy_limit=None):
        return [{"clause_id": "CL-001", "text": "Covered", "source_citation": "section 3.1"}]


class DummyReportService:
    def synthesize_report(self, detections, severity_summary, policy_findings, policy_context=None):
        return {
            "damage_table": [{"class": "dent", "severity": "Minor", "confidence": 0.82}],
            "severity_summary": {"overall": "Minor", "per_region": [{"class": "dent", "severity": "Minor"}]},
            "applicable_coverage": [{"summary": "Covered", "citations": [{"clause_id": "CL-001", "source": "section 3.1"}]}],
            "recommendation": "Approve",
            "confidence_score": 0.86,
            "next_steps": ["Monitor claim"],
        }


def test_fraud_agent_flags_name_mismatch():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="John Smith",
        claimed_amount="1000",
        incident_description="fender bender",
        severity_summary={"severity_score": 0.05},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
    )

    assert result["needs_investigation"] is True
    assert any("does not match" in signal for signal in result["signals"])


def test_fraud_agent_flags_expired_policy():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="1000",
        incident_description="fender bender",
        severity_summary={"severity_score": 0.05},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        incident_date="2026-08-09",
        policy_expiry_date="2026-01-01",
    )

    assert result["needs_investigation"] is True
    assert any("expired" in signal.lower() for signal in result["signals"])


def test_fraud_agent_flags_inactive_policy_status():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="1000",
        incident_description="fender bender",
        severity_summary={"severity_score": 0.05},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="cancelled",
    )

    assert result["needs_investigation"] is True
    assert any("not active" in signal.lower() for signal in result["signals"])


def test_fraud_agent_flags_vehicle_registration_mismatch():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="1000",
        incident_description="fender bender",
        severity_summary={"severity_score": 0.05},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        vehicle_no="MH-12-CD-9999",
        policy_vehicle_registration_no="MH-12-CD-5678",
    )

    assert result["needs_investigation"] is True
    assert result["rule_flags"]["vehicle_mismatch"] is True
    assert any("does not match" in signal and "vehicle" in signal.lower() for signal in result["signals"])


def test_fraud_agent_ignores_vehicle_registration_formatting_differences():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="500",
        incident_description="small scratch",
        severity_summary={"severity_score": 0.02},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        vehicle_no="mh12cd5678",
        policy_vehicle_registration_no="MH-12-CD-5678",
    )

    assert result["rule_flags"]["vehicle_mismatch"] is False


def test_fraud_agent_flags_cumulative_amount_over_policy_limit():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="1500",
        incident_description="fender bender",
        severity_summary={"severity_score": 0.05},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        already_claimed_amount="3000",
        policy_limit="4000",
    )

    assert result["needs_investigation"] is True
    assert any("exceeds the policy limit" in signal for signal in result["signals"])


def test_fraud_agent_no_signals_for_clean_claim():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="500",
        incident_description="small scratch",
        severity_summary={"severity_score": 0.02},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        incident_date="2026-08-09",
        policy_expiry_date="2028-01-01",
        already_claimed_amount="0",
        policy_limit="4000",
    )

    assert result["signals"] == []
    assert result["needs_investigation"] is False
    assert result["rule_flags"] == {
        "name_mismatch": False,
        "policy_inactive": False,
        "vehicle_mismatch": False,
        "exceeds_policy_limit": False,
    }


def test_fraud_agent_score_breakdown_traces_to_reported_score():
    """score_breakdown must be an explanation of fraud_score, not a
    separate computation -- its last step's running_total is the number
    actually returned as fraud_score (both rounded to 4dp before the
    outer 2dp round, so they can differ only in trailing precision)."""
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="John Smith",
        claimed_amount="1000",
        incident_description="fender bender",
        severity_summary={"severity_score": 0.05},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
    )

    breakdown = result["score_breakdown"]
    assert breakdown, "expected at least one breakdown step"
    assert round(breakdown[-1]["running_total"], 2) == result["fraud_score"]
    labels = [step["label"] for step in breakdown]
    assert "Claimant/policyholder name mismatch" in labels
    mismatch_step = next(step for step in breakdown if step["label"] == "Claimant/policyholder name mismatch")
    assert mismatch_step["triggered"] is True
    assert mismatch_step["delta"] > 0


def test_fraud_agent_score_breakdown_marks_untriggered_rules():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="500",
        incident_description="small scratch",
        severity_summary={"severity_score": 0.02},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        incident_date="2026-08-09",
        policy_expiry_date="2028-01-01",
        already_claimed_amount="0",
        policy_limit="4000",
    )

    breakdown = {step["label"]: step for step in result["score_breakdown"]}
    assert breakdown["Claimant/policyholder name mismatch"]["triggered"] is False
    assert breakdown["Claimant/policyholder name mismatch"]["delta"] == 0
    assert breakdown["Policy inactive or expired at incident date"]["triggered"] is False
    assert breakdown["Cumulative claimed amount exceeds policy limit"]["triggered"] is False


def test_fraud_agent_narrative_flags_add_bounded_score_and_stay_out_of_signals():
    """narrative_flags must move fraud_score and appear in score_breakdown
    (Langfuse-only, per how the orchestrator wires it), but must NOT be
    appended to `signals` -- that list is what the existing frontend
    fraud-factors panel already renders, and this feature is deliberately
    backend/observability-only, not a new UI surface."""
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="500",
        incident_description="ordinary fender bender",
        severity_summary={"severity_score": 0.02},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        narrative_flags=[
            {"quote": "needed the money fast", "concern": "financial pressure"},
            {"quote": "story kept changing", "concern": "inconsistent narrative"},
        ],
    )

    narrative_step = next(step for step in result["score_breakdown"] if step["label"] == "LLM-extracted narrative red flags")
    assert narrative_step["triggered"] is True
    assert narrative_step["delta"] == pytest.approx(0.1)
    assert result["signals"] == []


def test_fraud_agent_no_narrative_flags_marks_step_untriggered():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="500",
        incident_description="ordinary fender bender",
        severity_summary={"severity_score": 0.02},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
    )

    narrative_step = next(step for step in result["score_breakdown"] if step["label"] == "LLM-extracted narrative red flags")
    assert narrative_step["triggered"] is False
    assert narrative_step["delta"] == 0


def test_narrative_risk_analyzer_rejects_ungrounded_quotes():
    service = NarrativeRiskAnalyzerService()

    grounded = service._ground_flags(
        [
            {"quote": "the car caught fire suddenly", "concern": "arson"},
            {"quote": "this text was never in the description", "concern": "hallucinated"},
        ],
        "The claimant said the car caught fire suddenly near the parking lot.",
    )

    assert len(grounded) == 1
    assert grounded[0]["quote"] == "the car caught fire suddenly"


def test_narrative_risk_analyzer_short_circuits_on_empty_description():
    service = NarrativeRiskAnalyzerService()

    assert service.analyze("   ") == {"flags": [], "available": True, "reason": None}


def test_narrative_risk_analyzer_unavailable_without_api_key(monkeypatch):
    service = NarrativeRiskAnalyzerService()
    monkeypatch.setattr(service, "groq_api_key", None)

    result = service.analyze("Some incident description.")

    assert result == {"flags": [], "available": False, "reason": "GROQ_API_KEY is not configured"}


def test_report_consistency_service_flags_ungrounded_citation():
    service = ReportConsistencyService()
    report_json = {
        "recommendation": "Approve",
        "applicable_coverage": [{"summary": "x", "citations": [{"clause_id": "CL-999", "source": "s"}]}],
    }
    policy_findings = [{"clause_id": "CL-001", "clause_type": "coverage"}]

    result = service.check(report_json, policy_findings, {})

    citation_check = next(c for c in result["checks"] if c["rule"] == "citations_grounded")
    assert citation_check["passed"] is False
    assert result["all_passed"] is False


def test_report_consistency_service_flags_sub_limit_violation():
    service = ReportConsistencyService()
    report_json = {"recommendation": "Approve", "applicable_coverage": []}
    policy_findings = [{"clause_id": "POLICY-LIMIT-CHECK", "clause_type": "sub_limit", "status": "outside_policy_limit"}]

    result = service.check(report_json, policy_findings, {})

    check = next(c for c in result["checks"] if c["rule"] == "sub_limit_respected")
    assert check["passed"] is False


def test_report_consistency_service_flags_missing_coverage_basis():
    service = ReportConsistencyService()
    report_json = {"recommendation": "Approve", "applicable_coverage": []}

    result = service.check(report_json, [], {})

    check = next(c for c in result["checks"] if c["rule"] == "coverage_basis_present")
    assert check["passed"] is False


def test_report_consistency_service_flags_unreflected_fraud_signal():
    service = ReportConsistencyService()
    report_json = {
        "recommendation": "Approve",
        "applicable_coverage": [{"summary": "x", "citations": [{"clause_id": "CL-001", "source": "s"}]}],
    }
    policy_findings = [{"clause_id": "CL-001", "clause_type": "coverage"}]

    result = service.check(report_json, policy_findings, {"needs_investigation": True})

    check = next(c for c in result["checks"] if c["rule"] == "fraud_signals_reflected")
    assert check["passed"] is False
    assert result["all_passed"] is False


def test_report_consistency_service_all_pass_for_clean_report():
    service = ReportConsistencyService()
    report_json = {
        "recommendation": "Approve",
        "applicable_coverage": [{"summary": "x", "citations": [{"clause_id": "CL-001", "source": "s"}]}],
    }
    policy_findings = [{"clause_id": "CL-001", "clause_type": "coverage"}]

    result = service.check(report_json, policy_findings, {"needs_investigation": False})

    assert result["all_passed"] is True


def test_fraud_agent_rule_flags_reflect_expired_policy():
    service = FraudAgentService()
    result = service.assess_fraud_risk(
        claimant_name="Jane Doe",
        claimed_amount="500",
        incident_description="small scratch",
        severity_summary={"severity_score": 0.02},
        confidence_score=0.9,
        policy_holder_name="Jane Doe",
        policy_status="active",
        incident_date="2026-08-09",
        policy_expiry_date="2026-01-01",
    )

    assert result["rule_flags"]["policy_inactive"] is True


def test_orchestrator_assess_fraud_computes_cumulative_amount_from_policy_claims():
    """_assess_fraud must derive already_claimed_amount from the policy's
    other non-denied sibling claims (excluding the current claim and denied
    ones), not just the claim being analyzed."""
    from decimal import Decimal

    engine = _build_engine()

    policy = SimpleNamespace(
        policy_holder_name="Jane Doe",
        status="active",
        policy_expiry_date=None,
        policy_limit=Decimal("4000"),
        claims=[
            SimpleNamespace(claim_id="CLM-OTHER-1", status="approved", claimed_amount=Decimal("3000")),
            SimpleNamespace(claim_id="CLM-OTHER-2", status="denied", claimed_amount=Decimal("9000")),
            SimpleNamespace(claim_id="CLM-CURRENT", status="under review", claimed_amount=Decimal("1500")),
        ],
    )
    claim = SimpleNamespace(
        claim_id="CLM-CURRENT",
        claimant_name="Jane Doe",
        incident_description="fender bender",
        incident_date="2026-08-09",
        claimed_amount=Decimal("1500"),
    )

    state = {
        "claim": claim,
        "policy": policy,
        "severity_summary": {"severity_score": 0.05},
        "report_json": {"confidence_score": 0.9},
    }
    result_state = engine._assess_fraud(state)

    assessment = result_state["fraud_assessment"]
    # already_claimed (3000, denied claim excluded, current claim excluded)
    # + this claim (1500) = 4500 > policy_limit 4000.
    assert assessment["needs_investigation"] is True
    assert any("exceeds the policy limit" in signal for signal in assessment["signals"])


def test_orchestrator_overrides_approve_recommendation_when_policy_inactive():
    """An Approve recommendation is never valid when the policy wasn't active
    at the incident date -- the report-synthesis model has no visibility
    into policy status/expiry, so it can (and does) recommend Approve
    against a lapsed policy. That must be corrected deterministically."""
    engine = _build_engine()

    policy = SimpleNamespace(
        policy_holder_name="Jane Doe",
        status="inactive",
        policy_expiry_date=None,
        policy_limit=Decimal("4000"),
        claims=[],
    )
    claim = SimpleNamespace(
        claim_id="CLM-CURRENT",
        claimant_name="Jane Doe",
        incident_description="fender bender",
        incident_date="2026-08-09",
        claimed_amount=Decimal("500"),
    )

    state = {
        "claim": claim,
        "policy": policy,
        "severity_summary": {"severity_score": 0.02},
        "report_json": {"recommendation": "Approve", "confidence_score": 0.9},
    }
    result_state = engine._assess_fraud(state)

    report_json = result_state["report_json"]
    assert report_json["recommendation"] == "Investigate"
    assert report_json["original_recommendation"] == "Approve"
    assert report_json["recommendation_override_reason"]


def test_orchestrator_attaches_consistency_check_and_disabled_narrative_analysis():
    """_build_engine() never passes narrative_service, so it defaults to
    None -- this must degrade to a clearly-labeled disabled result rather
    than crashing or silently making a real Groq call in a unit test.
    report_json.consistency_check should reflect a clean, grounded report."""
    engine = _build_engine()

    policy = SimpleNamespace(
        policy_holder_name="Jane Doe",
        status="active",
        policy_expiry_date=None,
        policy_limit=Decimal("4000"),
        claims=[],
    )
    claim = SimpleNamespace(
        claim_id="CLM-CURRENT",
        claimant_name="Jane Doe",
        incident_description="fender bender",
        incident_date="2026-08-09",
        claimed_amount=Decimal("500"),
    )
    state = {
        "claim": claim,
        "policy": policy,
        "severity_summary": {"severity_score": 0.02},
        "policy_findings": [{"clause_id": "CL-001", "clause_type": "coverage"}],
        "report_json": {
            "recommendation": "Approve",
            "applicable_coverage": [{"summary": "x", "citations": [{"clause_id": "CL-001", "source": "s"}]}],
            "confidence_score": 0.9,
        },
    }

    result_state = engine._assess_fraud(state)

    assert result_state["fraud_assessment"]["narrative_analysis"] == {
        "flags": [], "available": False, "reason": "Narrative risk analysis not configured",
    }
    assert result_state["report_json"]["consistency_check"]["all_passed"] is True


class SaliencyAwareDamageService:
    def detect_from_path(self, image_path):
        return [{"class_name": "dent", "bbox": [0, 0, 20, 20], "mask_polygon": None, "confidence": 0.82}]

    def get_image_dimensions(self, image_path):
        return (100, 100)

    def explain_detection(self, image_path, detection, grid_size=4):
        return {
            "class_name": detection["class_name"],
            "bbox": detection["bbox"],
            "grid_size": grid_size,
            "method": "occlusion_sensitivity",
            "baseline_confidence": detection["confidence"],
            "importance": [[1.0]],
            "peak_cell": {"row": 0, "col": 0},
            "peak_confidence_drop": 0.5,
        }


def test_orchestrator_attaches_saliency_onto_detection_for_frontend_consumption():
    """Saliency must be attached to the same detection dict that's already
    persisted to AnalysisResult.detections and returned by the claim detail
    API -- a Langfuse-only span would never be reachable from the app UI,
    so the frontend has no other way to receive this data."""
    engine = _build_engine(damage_service=SaliencyAwareDamageService())
    claim = SimpleNamespace(photos=[SimpleNamespace(file_path="fake.jpg")])
    state = {"claim": claim, "trace": None}

    result_state = engine._detect_damage(state)

    assert result_state["detections"][0]["saliency"]["method"] == "occlusion_sensitivity"
    assert result_state["detections"][0]["saliency"]["peak_cell"] == {"row": 0, "col": 0}


def test_orchestrator_does_not_override_recommendation_when_policy_active():
    engine = _build_engine()

    policy = SimpleNamespace(
        policy_holder_name="Jane Doe",
        status="active",
        policy_expiry_date=None,
        policy_limit=Decimal("4000"),
        claims=[],
    )
    claim = SimpleNamespace(
        claim_id="CLM-CURRENT",
        claimant_name="Jane Doe",
        incident_description="fender bender",
        incident_date="2026-08-09",
        claimed_amount=Decimal("500"),
    )

    state = {
        "claim": claim,
        "policy": policy,
        "severity_summary": {"severity_score": 0.02},
        "report_json": {"recommendation": "Approve", "confidence_score": 0.9},
    }
    result_state = engine._assess_fraud(state)

    report_json = result_state["report_json"]
    assert report_json["recommendation"] == "Approve"
    assert "original_recommendation" not in report_json
