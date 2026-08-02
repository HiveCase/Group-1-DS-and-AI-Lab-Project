"""
Graph-logic tests: routing, interrupt/resume, faithfulness rules, severity.

These run entirely offline. The YOLO checkpoint and the Groq call are
stubbed, because what is under test here is the *control flow* -- does an
escalation actually pause the run, does a resume continue without re-running
upstream nodes, does a grounding failure block. Testing that against a live
model would make the suite slow, cost money, and (since the model is
non-deterministic) flaky about things that have nothing to do with the graph.

Run: python -m pytest tests/test_claim_graph.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agents.graph import build_graph
from src.evaluation.faithfulness import check_report
from src.schemas import make_detection, new_claim_state, severity_from_area_ratio


# Stubs

class StubDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def detect(self, image_path):
        self.calls += 1
        return self.detections


class StubClauseRetriever:
    """Returns one coverage and one exclusion clause for any damage class,
    unless configured to report no coverage."""

    def __init__(self, coverage=True):
        self.coverage = coverage

    def get_clauses(self, damage_class):
        cov = [] if not self.coverage else [{
            "chunk_id": "chunk_00001", "text": "We will indemnify accidental damage.",
            "heading": "SECTION I", "clause_type": "coverage",
            "doc_id": "user_test", "score": 0.06,
        }]
        return {
            "coverage": cov,
            "exclusion_or_condition": [{
                "chunk_id": "chunk_00002", "text": "Wear and tear is excluded.",
                "heading": "EXCLUSIONS", "clause_type": "exclusion",
                "doc_id": "user_test", "score": 0.05,
            }],
            "coverage_clause_found": bool(cov),
        }


class StubResources:
    def __init__(self, detections, report, coverage=True):
        self.detector = StubDetector(detections)
        self.llm_client = None
        self._retriever = StubClauseRetriever(coverage)
        self.report = report
        self.llm_calls = 0

    def get_clause_retriever(self, user_id):
        return self._retriever


@pytest.fixture
def patched_llm(monkeypatch):
    """Patch the report agent so no network call happens."""
    def _apply(resources):
        def fake(context_bundle, model="stub-model", client=None, max_retries=1):
            resources.llm_calls += 1
            return {"ok": True, "parsed": resources.report, "raw": "",
                    "error": None, "model": "stub-model"}
        monkeypatch.setattr("src.agents.report_agent.generate_report", fake)
        # faithfulness reads the user's chunk TSV; there is no real user here.
        monkeypatch.setattr("src.evaluation.faithfulness.load_chunk_damage_classes",
                            lambda user_id: {})
    return _apply


def _good_report(damage_class="dent", verdict="covered", escalate=False):
    return {
        "claim_id": "C1", "policy_doc_id": "user_test",
        "items": [{"damage_class": damage_class, "verdict": verdict,
                   "rationale": "Grounded in the coverage clause.",
                   "cited_chunk_ids": ["chunk_00001"]}],
        "overall_recommendation": verdict,
        "escalate_to_human": escalate, "escalation_reason": None,
    }


def _run(resources, patched_llm, state=None):
    patched_llm(resources)
    graph = build_graph(resources).compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "user_test:C1"}}
    state = state or new_claim_state("C1", "test", "photo.jpg", "A dent appeared.")
    graph.invoke(state, config=config)
    return graph, config


# Severity proxy

@pytest.mark.parametrize("area_ratio,expected", [
    (0.005, "minor"), (0.02, "minor"), (0.05, "moderate"),
    (0.08, "moderate"), (0.5, "severe"),
])
def test_severity_bins(area_ratio, expected):
    assert severity_from_area_ratio(area_ratio) == expected


def test_detection_derives_area_from_bbox():
    """area_ratio must come from the bbox (w*h), since the severity bins were
    calibrated against bbox area, not segmentation-mask area."""
    d = make_detection("dent", area_ratio=0.04, confidence=0.9)
    assert d.area_ratio == pytest.approx(0.04, abs=1e-4)
    assert d.severity == "moderate"


# Routing

def test_confident_claim_completes_without_review(patched_llm):
    resources = StubResources([make_detection("dent", 0.04, 0.95)], _good_report())
    graph, config = _run(resources, patched_llm)

    snapshot = graph.get_state(config)
    assert not snapshot.next, "claim should have run to completion"
    assert snapshot.values["status"] == "completed"
    assert snapshot.values["final_report"]["outcome"] == "auto_assessed"


def test_low_confidence_detection_pauses_for_review(patched_llm):
    resources = StubResources([make_detection("dent", 0.04, 0.30)], _good_report(escalate=True))
    graph, config = _run(resources, patched_llm)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",), "should be paused at human_review"
    assert snapshot.values["needs_human_review"] is True
    assert any("Low-confidence" in r for r in snapshot.values["escalation_reasons"])


def test_no_detections_pauses_rather_than_denying(patched_llm):
    """An empty detection set must not become 'nothing is covered'."""
    resources = StubResources([], _good_report())
    graph, config = _run(resources, patched_llm)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert any("No damage detected" in r for r in snapshot.values["escalation_reasons"])
    assert resources.llm_calls == 0, "must not call the LLM with nothing to assess"


def test_missing_coverage_clause_pauses(patched_llm):
    resources = StubResources([make_detection("dent", 0.04, 0.95)],
                              _good_report(escalate=True), coverage=False)
    graph, config = _run(resources, patched_llm)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert any("No coverage clause" in r for r in snapshot.values["escalation_reasons"])


# Interrupt / resume

def test_resume_finalizes_without_rerunning_upstream(patched_llm):
    resources = StubResources([make_detection("dent", 0.04, 0.30)], _good_report(escalate=True))
    graph, config = _run(resources, patched_llm)

    assert graph.get_state(config).next == ("human_review",)
    detect_calls_before = resources.detector.calls
    llm_calls_before = resources.llm_calls

    graph.invoke(Command(resume={"decision": "approve", "reviewer": "kamal"}), config=config)

    snapshot = graph.get_state(config)
    assert not snapshot.next
    assert snapshot.values["status"] == "completed"
    assert snapshot.values["final_report"]["outcome"] == "approved_by_reviewer"
    # The point of checkpointing: resuming re-runs neither detection nor the LLM.
    assert resources.detector.calls == detect_calls_before
    assert resources.llm_calls == llm_calls_before


def test_reviewer_rejection_overrides_model_verdict(patched_llm):
    resources = StubResources([make_detection("dent", 0.04, 0.30)],
                              _good_report(verdict="covered", escalate=True))
    graph, config = _run(resources, patched_llm)

    graph.invoke(Command(resume={"decision": "reject", "reviewer": "kamal"}), config=config)
    final = graph.get_state(config).values["final_report"]
    assert final["outcome"] == "rejected_by_reviewer"


def test_reviewer_decision_outranks_missing_report(patched_llm):
    """A claim with nothing detected still reaches a reviewer, and their
    decision settles it. Checking 'is there a model report?' before 'did a
    human decide?' marked these `failed` even after an explicit approval."""
    resources = StubResources([], _good_report())
    graph, config = _run(resources, patched_llm)
    assert graph.get_state(config).next == ("human_review",)
    assert resources.llm_calls == 0

    graph.invoke(Command(resume={"decision": "approve", "reviewer": "kamal",
                                 "notes": "Damage confirmed on inspection."}),
                 config=config)

    values = graph.get_state(config).values
    assert values["status"] == "completed"
    assert values["final_report"]["outcome"] == "approved_by_reviewer"


def test_unreviewed_claim_without_report_is_failed():
    """The other side of the same rule: no report AND no human decision must
    never be reported as a successful assessment."""
    from src.agents.nodes import finalize

    state = new_claim_state("C1", "test", "photo.jpg", "narrative")
    state["report"] = None
    state["review_decision"] = None

    result = finalize(state, resources=None)
    assert result["status"] == "failed"
    assert result["final_report"]["outcome"] == "no_report_produced"


def test_reviewer_override_replaces_report(patched_llm):
    resources = StubResources([make_detection("dent", 0.04, 0.30)], _good_report(escalate=True))
    graph, config = _run(resources, patched_llm)

    corrected = _good_report(verdict="excluded")
    graph.invoke(
        Command(resume={"decision": "override", "reviewer": "kamal",
                        "overridden_report": corrected}),
        config=config,
    )
    final = graph.get_state(config).values["final_report"]
    assert final["report"]["overall_recommendation"] == "excluded"


# Faithfulness rules

def _payload(needs_review=False):
    return {
        "claim_id": "C1",
        "detections": [{"class_name": "dent"}],
        "policy": {"clauses": {"dent": {
            "coverage": [{"chunk_id": "chunk_00001", "text": "We will indemnify."}],
            "exclusion_or_condition": [{"chunk_id": "chunk_00002", "text": "Wear excluded."}],
        }}},
        "escalation": {"needs_human_review": needs_review},
    }


def test_citation_never_offered_is_hard_failure():
    report = _good_report()
    report["items"][0]["cited_chunk_ids"] = ["chunk_09999"]
    result = check_report(_payload(), report)
    assert result["passed"] is False
    assert any("never offered" in f for f in result["hard_failures"])


def test_covered_verdict_without_coverage_citation_is_hard_failure():
    report = _good_report(verdict="covered")
    report["items"][0]["cited_chunk_ids"] = ["chunk_00002"]  # exclusion only
    result = check_report(_payload(), report)
    assert result["passed"] is False
    assert any("cites no coverage clause" in f for f in result["hard_failures"])


def test_suppressing_mandated_escalation_is_hard_failure():
    report = _good_report(escalate=False)
    result = check_report(_payload(needs_review=True), report)
    assert result["passed"] is False
    assert any("mandated escalation suppressed" in f for f in result["hard_failures"])


def test_model_initiated_escalation_is_soft_not_hard():
    """The prompt's rule 4 is one-directional: the model volunteering an
    escalation is caution to honour, not a grounding violation."""
    report = _good_report(escalate=True)
    result = check_report(_payload(needs_review=False), report)
    assert result["passed"] is True
    assert any(f["type"] == "model_initiated_escalation" for f in result["soft_flags"])


def test_negative_verdict_on_coverage_clause_is_soft_not_hard():
    """An 'excluded' verdict resting on a coverage clause's own condition is
    legitimate reasoning, so it flags rather than blocks."""
    report = _good_report(verdict="excluded")
    report["items"][0]["cited_chunk_ids"] = ["chunk_00001"]  # coverage only
    result = check_report(_payload(), report)
    assert result["passed"] is True
    assert any(f["type"] == "negative_verdict_on_coverage_clause_only"
               for f in result["soft_flags"])


def test_fabricated_currency_figure_is_hard_failure():
    report = _good_report()
    report["items"][0]["rationale"] = "The claim is payable up to Rs. 50,000."
    result = check_report(_payload(), report)
    assert result["passed"] is False
    assert any("currency figure" in f for f in result["hard_failures"])


def test_missing_verdict_for_detected_class_is_hard_failure():
    payload = _payload()
    payload["detections"].append({"class_name": "scratch"})
    result = check_report(payload, _good_report())
    assert result["passed"] is False
    assert any("no verdict returned" in f for f in result["hard_failures"])


def test_reviewer_override_outcome_is_labelled_distinctly(patched_llm):
    resources = StubResources([make_detection("dent", 0.04, 0.30)], _good_report(escalate=True))
    graph, config = _run(resources, patched_llm)
    graph.invoke(
        Command(resume={"decision": "override", "reviewer": "kamal",
                        "overridden_report": _good_report(verdict="excluded")}),
        config=config,
    )
    assert graph.get_state(config).values["final_report"]["outcome"] == "overridden_by_reviewer"


def test_reviewed_claim_is_never_labelled_auto_assessed():
    """An audit record must not simultaneously say a human reviewed the claim
    and that the machine assessed it unattended."""
    from src.agents.nodes import finalize

    state = new_claim_state("C1", "test", "photo.jpg", "narrative")
    state["report"] = {"items": [], "overall_recommendation": "covered"}
    state["review_decision"] = {"reviewer": "kamal", "notes": "no verb"}

    final = finalize(state, resources=None)["final_report"]
    assert final["human_reviewed"] is True
    assert final["outcome"] != "auto_assessed"


@pytest.mark.parametrize("bad", [
    {"reviewer": "kamal"},                      # no verb
    {"decision": "maybe", "reviewer": "kamal"}, # unknown verb
    {"decision": "override"},                   # override without a report
])
def test_resume_rejects_invalid_decisions(bad, monkeypatch):
    import src.agents.graph as graph_mod

    # Reject before any graph/checkpointer work happens.
    monkeypatch.setattr(graph_mod, "get_compiled_graph",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("validation must happen before graph access")))
    with pytest.raises(ValueError):
        graph_mod.resume_claim("test", "C1", bad)


def test_grounding_failure_routes_to_human_review(patched_llm):
    """End-to-end: a report citing a chunk it was never shown must escalate."""
    bad = _good_report()
    bad["items"][0]["cited_chunk_ids"] = ["chunk_09999"]
    resources = StubResources([make_detection("dent", 0.04, 0.95)], bad)
    graph, config = _run(resources, patched_llm)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert snapshot.values["faithfulness"]["passed"] is False
