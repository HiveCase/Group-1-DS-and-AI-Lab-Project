"""
Mechanical grounding checks on a generated report.

Every check here is deterministic and reproducible from the payload + report
JSON alone -- no LLM-judge pass. That is the point: this is the component
that decides whether the model's claims about the policy are actually
supported, so it cannot itself depend on a model's judgement.

Architectural change from the scripts/ version: there, these checks were an
offline eval that scored two models after the fact and had no effect on any
claim. Here the same checks run inline as a graph node, and a hard failure
routes the claim to human review. A citation to a chunk the model was never
shown now stops a claim instead of appearing in a summary table nobody
reads.

Hard vs. soft is a deliberate distinction:
  - HARD (blocks): the report contradicts its own evidence, cites clauses it
    was never given, omits a detected damage class, or states a currency
    figure the prompt forbids and no clause contains.
  - SOFT (flagged, does not block): patterns that are usually legitimate but
    worth a reviewer's eye. Chief among them, an "excluded"/"conditional"
    verdict citing only a coverage clause -- that is often *correct*
    reasoning, e.g. a tyre clause that covers damage "only when the vehicle
    has also been damaged" (condition unmet -> excluded, citing the coverage
    clause itself), or crack coverage enumerating "bumpers, door panels,
    trims" where a lamp-housing crack falls outside the enumerated scope.
    Treating those as failures produced false positives, so they surface
    rather than block.
"""
import csv
import logging
import re

from src.config import VALID_VERDICTS
from src.retrieval.policy_store import chunks_tsv_path

log = logging.getLogger(__name__)

CURRENCY_RE = re.compile(r"(₹|Rs\.?\s?\d|INR\s?\d)", re.IGNORECASE)

REQUIRED_KEYS = {"claim_id", "policy_doc_id", "items", "overall_recommendation",
                 "escalate_to_human"}


def load_chunk_damage_classes(user_id: str) -> dict:
    """chunk_id -> damage_classes, for this user's own policy only.

    Chunk ids are unique only within a user's own index (each per-user
    collection restarts numbering at chunk_00000), so this must be resolved
    per user -- a shared lookup would silently attribute one user's chunk
    metadata to another's identically-named chunk.
    """
    path = chunks_tsv_path(user_id)
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["chunk_id"]] = [c for c in row["damage_classes"].split(",") if c]
    return out


def check_report(payload: dict, parsed: dict, chunk_classes: dict = None) -> dict:
    """Run all checks. Returns a result dict with `passed` (no hard
    failures), `hard_failures`, and `soft_flags`."""
    chunk_classes = chunk_classes or {}
    hard, soft = [], []

    if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed.keys()):
        missing = REQUIRED_KEYS - set(parsed.keys()) if isinstance(parsed, dict) else REQUIRED_KEYS
        return {
            "passed": False,
            "schema_valid": False,
            "hard_failures": [f"report missing required keys: {sorted(missing)}"],
            "soft_flags": [],
        }

    items = parsed["items"] if isinstance(parsed["items"], list) else []
    detected = {d["class_name"] for d in payload.get("detections", [])}
    reported = {it.get("damage_class") for it in items}

    if not detected.issubset(reported):
        hard.append(f"no verdict returned for detected class(es): {sorted(detected - reported)}")

    total_citations = valid_citations = 0
    multi_class_citations = []

    for it in items:
        cls = it.get("damage_class")
        verdict = it.get("verdict")
        if verdict not in VALID_VERDICTS:
            hard.append(f"invalid verdict '{verdict}' for {cls}")

        bucket = payload["policy"]["clauses"].get(cls, {"coverage": [], "exclusion_or_condition": []})
        offered_coverage = {c["chunk_id"] for c in bucket["coverage"]}
        offered_exclusion = {c["chunk_id"] for c in bucket["exclusion_or_condition"]}
        offered_all = offered_coverage | offered_exclusion

        cited = it.get("cited_chunk_ids") or []
        for cid in cited:
            total_citations += 1
            if cid in offered_all:
                valid_citations += 1
            else:
                # Citing a real corpus chunk the model was never shown is
                # exactly as much a grounding failure as inventing an id.
                hard.append(f"citation '{cid}' for {cls} was never offered in the payload")
            if len(chunk_classes.get(cid, [])) > 1:
                multi_class_citations.append(cid)

        if verdict == "covered" and not (set(cited) & offered_coverage):
            hard.append(f"'covered' verdict for {cls} cites no coverage clause")

        if verdict in {"excluded", "conditional"} and not (set(cited) & offered_exclusion):
            soft.append({"type": "negative_verdict_on_coverage_clause_only",
                         "damage_class": cls, "verdict": verdict})

        rationale = it.get("rationale") or ""
        if CURRENCY_RE.search(rationale):
            offered_text = " ".join(c["text"] for c in bucket["coverage"] + bucket["exclusion_or_condition"])
            if not CURRENCY_RE.search(offered_text):
                hard.append(f"rationale for {cls} states a currency figure absent from every clause shown")

    overall = parsed.get("overall_recommendation") or ""
    if CURRENCY_RE.search(overall):
        hard.append("overall_recommendation states a currency figure")

    # Escalation consistency is deliberately ONE-directional, matching the
    # prompt's rule 4 ("if needs_human_review is true, set
    # escalate_to_human=true"). Suppressing a mandated escalation is a
    # violation. Raising one the payload didn't demand is not: the model
    # noticing that the clauses it received don't actually support a
    # confident answer is exactly the behaviour this pipeline wants, and a
    # bidirectional equality check punishes it. That symmetric check was
    # inherited from the scripts/ eval, where it never fired only because
    # the 10 synthetic payloads happened to agree with the model every time;
    # the first real claim run against a real policy tripped it immediately.
    payload_needs_review = bool(payload["escalation"]["needs_human_review"])
    model_escalates = bool(parsed.get("escalate_to_human"))
    if payload_needs_review and not model_escalates:
        hard.append(
            "escalate_to_human=False despite payload needs_human_review=True "
            "(mandated escalation suppressed)"
        )
    elif model_escalates and not payload_needs_review:
        soft.append({
            "type": "model_initiated_escalation",
            "reason": parsed.get("escalation_reason"),
        })

    for cid in sorted(set(multi_class_citations)):
        # Proxy for "this chunk is probably a merged/garbled PDF table row
        # spanning multiple coverage topics" -- a confirmed case existed in
        # the original corpus where one chunk glued a glass-depreciation
        # table cell onto the following tyre row's condition. Not evidence
        # the model erred, so it never blocks.
        soft.append({"type": "multi_class_chunk_citation", "chunk_id": cid})

    return {
        "passed": len(hard) == 0,
        "schema_valid": True,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "citation_validity_rate": round(valid_citations / total_citations, 3) if total_citations else None,
        "hard_failures": hard,
        "soft_flags": soft,
    }
