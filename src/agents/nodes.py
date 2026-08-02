"""
The graph's nodes. Each takes ClaimState and returns only the keys it
changes; LangGraph merges them into the checkpointed state.

Nodes hold no state of their own. Expensive, reusable objects (the YOLO
checkpoint, the embedding model) are injected via a Resources handle built
once in src/agents/graph.py, because a node that constructs its own detector
would reload ~24MB of weights on every claim.

Error handling convention: a node that fails in a recoverable way records
the reason, sets needs_human_review, and returns normally. Raising would
abort the run and lose the checkpoint; the entire point of routing to a
human is that an unassessable claim is a valid outcome, not a crash.
"""
import logging

from langgraph.types import interrupt

from src.config import ESCALATION_CONFIDENCE_THRESHOLD, REPORT_MODEL
from src.evaluation.faithfulness import check_report, load_chunk_damage_classes
from src.retrieval.clause_retriever import ClauseRetriever, build_context_bundle
from src.schemas import ClaimState

log = logging.getLogger(__name__)


def detect_damage(state: ClaimState, resources) -> dict:
    """Run the segmentation model on the claim photo."""
    try:
        detections = resources.detector.detect(state["image_path"])
    except Exception as e:
        log.error("Detection failed for %s: %s", state["claim_id"], e)
        return {
            "detections": [],
            "errors": state.get("errors", []) + [f"detection failed: {e}"],
            "needs_human_review": True,
            "escalation_reasons": state.get("escalation_reasons", []) + [
                f"Damage detection failed: {e}"],
        }

    reasons = list(state.get("escalation_reasons", []))
    needs_review = state.get("needs_human_review", False)

    if not detections:
        # No damage found is not "nothing is covered" -- it could be a photo
        # problem, an angle problem, or damage outside the 6 trained classes.
        # Only a human can tell those apart, so the claim stops here.
        needs_review = True
        reasons.append("No damage detected in the submitted photo.")

    low_conf = [d.class_name for d in detections
                if d.confidence < ESCALATION_CONFIDENCE_THRESHOLD]
    if low_conf:
        needs_review = True
        reasons.append(
            f"Low-confidence detection(s) below {ESCALATION_CONFIDENCE_THRESHOLD}: "
            f"{sorted(set(low_conf))}")

    return {
        "detections": [d.to_dict() for d in detections],
        "needs_human_review": needs_review,
        "escalation_reasons": reasons,
    }


def retrieve_clauses(state: ClaimState, resources) -> dict:
    """Per damage class, pull coverage and exclusion clauses from this
    user's own policy index."""
    damage_classes = sorted({d["class_name"] for d in state.get("detections", [])})
    if not damage_classes:
        return {"clauses_by_class": {}, "missing_coverage_for": []}

    retriever = resources.get_clause_retriever(state["user_id"])

    clauses_by_class = {}
    missing = []
    for cls in damage_classes:
        clauses_by_class[cls] = retriever.get_clauses(cls)
        if not clauses_by_class[cls]["coverage_clause_found"]:
            missing.append(cls)

    reasons = list(state.get("escalation_reasons", []))
    needs_review = state.get("needs_human_review", False)
    if missing:
        needs_review = True
        reasons.append(f"No coverage clause found in the uploaded policy for: {missing}")

    return {
        "clauses_by_class": clauses_by_class,
        "missing_coverage_for": missing,
        "needs_human_review": needs_review,
        "escalation_reasons": reasons,
    }


def build_bundle(state: ClaimState, resources) -> dict:
    """Assemble the exact JSON the LLM will see."""
    low_conf = [d["class_name"] for d in state.get("detections", [])
                if d["confidence"] < ESCALATION_CONFIDENCE_THRESHOLD]

    bundle = build_context_bundle(
        claim_id=state["claim_id"],
        user_id=state["user_id"],
        incident_narrative=state["incident_narrative"],
        detections=state.get("detections", []),
        clauses_by_class=state.get("clauses_by_class", {}),
        escalation={
            "low_confidence_detections": low_conf,
            "missing_coverage_clause_for": state.get("missing_coverage_for", []),
            "needs_human_review": bool(state.get("needs_human_review", False)),
        },
    )
    return {"context_bundle": bundle}


def generate_report(state: ClaimState, resources) -> dict:
    """Single LLM call. Skipped entirely when there is nothing to assess."""
    if not state.get("detections"):
        return {
            "report": None,
            "report_model": REPORT_MODEL,
            "report_error": "skipped: no detections to assess",
        }

    from src.agents.report_agent import generate_report as call_llm

    result = call_llm(state["context_bundle"], client=resources.llm_client)

    if not result["ok"]:
        return {
            "report": None,
            "report_model": result["model"],
            "report_error": result["error"],
            "needs_human_review": True,
            "escalation_reasons": state.get("escalation_reasons", []) + [
                f"Report generation failed: {result['error']}"],
        }

    return {"report": result["parsed"], "report_model": result["model"], "report_error": None}


def check_faithfulness(state: ClaimState, resources) -> dict:
    """Verify the report is actually grounded in the clauses it was shown.

    This is where the architecture differs most from the scripts/ pipeline:
    there, these checks scored models offline and changed nothing. Here a
    hard failure escalates the claim.
    """
    if state.get("report") is None:
        return {"faithfulness": {"passed": False, "skipped": True,
                                 "hard_failures": ["no report to check"],
                                 "soft_flags": []}}

    chunk_classes = load_chunk_damage_classes(state["user_id"])
    result = check_report(state["context_bundle"], state["report"], chunk_classes)

    reasons = list(state.get("escalation_reasons", []))
    needs_review = state.get("needs_human_review", False)
    if not result["passed"]:
        needs_review = True
        reasons.append(f"Report failed grounding checks: {result['hard_failures']}")

    # The model may escalate on its own initiative even when nothing upstream
    # flagged the claim -- typically because the clauses it was given don't
    # actually let it decide. That judgement is honoured: it routes to a
    # human rather than being recorded and ignored.
    for flag in result.get("soft_flags", []):
        if flag.get("type") == "model_initiated_escalation":
            needs_review = True
            reasons.append(f"Model requested human review: {flag.get('reason')}")

    return {
        "faithfulness": result,
        "needs_human_review": needs_review,
        "escalation_reasons": reasons,
    }


def human_review(state: ClaimState, resources) -> dict:
    """Pause the run until a reviewer resumes it.

    interrupt() halts execution here and persists the checkpoint. The claim
    can sit in this state indefinitely -- across process restarts -- and
    resume exactly here when src/cli.py resume-claim supplies a decision.
    Nothing before this node re-runs on resume.
    """
    decision = interrupt({
        "claim_id": state["claim_id"],
        "user_id": state["user_id"],
        "reason": "This claim requires human review before it can be finalised.",
        "escalation_reasons": state.get("escalation_reasons", []),
        "detections": state.get("detections", []),
        "draft_report": state.get("report"),
        "faithfulness": state.get("faithfulness", {}),
        "expected_response": {
            "decision": "approve | reject | override",
            "reviewer": "<name or id>",
            "notes": "<optional>",
            "overridden_report": "<full report JSON, only when decision=override>",
        },
    })

    if isinstance(decision, str):
        decision = {"decision": decision}

    return {"review_decision": decision}


def finalize(state: ClaimState, resources) -> dict:
    """Produce the claim's terminal record.

    A reviewer's decision outranks the model: `override` replaces the report
    body outright, `reject` marks it rejected regardless of what the model
    concluded. A claim that reached here without review keeps the model's
    report as-is.
    """
    decision = state.get("review_decision") or {}
    action = (decision.get("decision") or "").lower()

    report = state.get("report")
    if action == "override" and decision.get("overridden_report"):
        report = decision["overridden_report"]

    final = {
        "claim_id": state["claim_id"],
        "user_id": state["user_id"],
        "image_path": state.get("image_path"),
        "incident_narrative": state.get("incident_narrative"),
        "detections": state.get("detections", []),
        "report": report,
        "report_model": state.get("report_model"),
        "report_error": state.get("report_error"),
        "faithfulness": state.get("faithfulness", {}),
        "human_reviewed": bool(decision),
        "review_decision": decision or None,
        "escalation_reasons": state.get("escalation_reasons", []),
        "errors": state.get("errors", []),
    }

    # A reviewer's decision is the authority and is checked FIRST, before the
    # presence of a model report. The ordering matters: a claim can reach
    # review with no report at all (nothing detected, or generation failed),
    # and a human who inspects it and decides is precisely the mechanism that
    # is supposed to resolve that. Checking "is there a report?" first marked
    # such claims `failed` even when a reviewer had explicitly approved them
    # -- discarding the human judgement the escalation existed to obtain.
    if action in {"approve", "reject", "override"}:
        status = "completed"
        final["outcome"] = {
            "approve": "approved_by_reviewer",
            "reject": "rejected_by_reviewer",
            "override": "overridden_by_reviewer",
        }[action]
    elif decision:
        # A reviewer acted but the verb wasn't recognised. resume_claim()
        # validates this, so reaching here means state was written by some
        # other path -- record it as needing attention rather than silently
        # labelling it "auto_assessed", which would contradict the
        # human_reviewed=True on this very record.
        status = "failed"
        final["outcome"] = "unrecognised_review_decision"
    elif report is None:
        # No assessment and nobody decided -- never silently call this done.
        status = "failed"
        final["outcome"] = "no_report_produced"
    else:
        status = "completed"
        final["outcome"] = "auto_assessed"

    return {"status": status, "final_report": final}


def route_after_faithfulness(state: ClaimState) -> str:
    """Conditional edge: escalate or finalise."""
    return "human_review" if state.get("needs_human_review") else "finalize"
