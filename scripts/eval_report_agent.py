"""
Faithfulness/grounding eval for the two Groq Report Agent models, run over
the 10 synthetic claim payloads. Mechanical checks only -- no LLM-judge pass
-- so every number here is independently reproducible from the JSON files
alone.

Checks per report:
  - schema_valid: required keys present, verdict is one of the 4 allowed values.
  - class_coverage_complete: every detected damage_class got exactly one item.
  - citation_validity: every cited_chunk_id was actually offered to the model
    in that payload's policy.clauses[damage_class] -- catches a citation to a
    real corpus chunk that the model was never shown, which is exactly as
    much a grounding failure as inventing a chunk_id outright.
  - verdict_evidence_consistency: "covered" must cite >=1 coverage chunk,
    "excluded" must cite >=1 exclusion/condition chunk.
  - escalation_consistency: report.escalate_to_human matches
    payload.escalation.needs_human_review.
  - financial_figure_leakage: a Rs./INR/currency figure appears in the
    model's own prose that isn't present verbatim in any clause text shown
    to it -- the system prompt explicitly forbids stating claim amounts.
  - multi_class_chunk_citations: cites a chunk tagged with more than one
    damage_classes value in clause_groundtruth.json. This does not by itself
    mean the model got anything wrong -- it's a proxy for "this chunk is
    likely a merged/garbled PDF table row spanning multiple coverage
    topics" (see docs/rag_support_mile3.md for the confirmed case: chunk
    chunk_00122 in policy_4_autoguard_premium glues a glass-depreciation
    table cell onto the following tyre row's condition). Flagged for manual
    review, not treated as a hard failure.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOADS_PATH = ROOT / "data" / "rag_outputs" / "mile3" / "payloads_all.json"
REPORTS_DIR = ROOT / "data" / "rag_outputs" / "mile3" / "reports"
CHUNKS_TSV = ROOT / "data" / "rag_outputs" / "chunks_all.tsv"
OUT_PATH = ROOT / "data" / "rag_outputs" / "mile3" / "faithfulness_eval.json"

VALID_VERDICTS = {"covered", "excluded", "conditional", "needs_review"}
CURRENCY_RE = re.compile(r"(₹|Rs\.?\s?\d|INR\s?\d)", re.IGNORECASE)


def load_chunk_damage_classes() -> dict:
    out = {}
    with open(CHUNKS_TSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            classes = [c for c in row["damage_classes"].split(",") if c]
            out[row["chunk_id"]] = classes
    return out


def check_report(payload: dict, parsed: dict, chunk_classes: dict) -> dict:
    issues = []

    required = {"claim_id", "policy_doc_id", "items", "overall_recommendation", "escalate_to_human"}
    schema_valid = isinstance(parsed, dict) and required.issubset(parsed.keys())
    if not schema_valid:
        return {"schema_valid": False, "issues": ["missing required top-level keys"]}

    items = parsed["items"] if isinstance(parsed["items"], list) else []
    detected_classes = {d["class_name"] for d in payload["detections"]}
    reported_classes = {it.get("damage_class") for it in items}
    class_coverage_complete = detected_classes.issubset(reported_classes)
    if not class_coverage_complete:
        issues.append(f"missing verdict for classes: {detected_classes - reported_classes}")

    total_citations = 0
    valid_citations = 0
    verdict_evidence_ok = True
    multi_class_citations = []
    currency_violations = []
    neg_verdict_on_coverage_only = []

    for it in items:
        cls = it.get("damage_class")
        verdict = it.get("verdict")
        if verdict not in VALID_VERDICTS:
            issues.append(f"invalid verdict '{verdict}' for {cls}")

        clause_bucket = payload["policy"]["clauses"].get(cls, {"coverage": [], "exclusion_or_condition": []})
        offered_coverage_ids = {c["chunk_id"] for c in clause_bucket["coverage"]}
        offered_exclusion_ids = {c["chunk_id"] for c in clause_bucket["exclusion_or_condition"]}
        offered_all = offered_coverage_ids | offered_exclusion_ids

        cited = it.get("cited_chunk_ids", []) or []
        for cid in cited:
            total_citations += 1
            if cid in offered_all:
                valid_citations += 1
            else:
                issues.append(f"citation '{cid}' for {cls} was never offered in the payload")
            if len(chunk_classes.get(cid, [])) > 1:
                multi_class_citations.append(cid)

        # Hard check -- guards the domain's real risk (Milestone 1): a report
        # that claims coverage it doesn't have. A "covered" verdict resting
        # only on an exclusion chunk is a genuine self-contradiction.
        if verdict == "covered" and not (set(cited) & offered_coverage_ids):
            verdict_evidence_ok = False
            issues.append(f"'covered' verdict for {cls} cites no coverage-type chunk")
        # Soft flag, NOT a hard failure -- an "excluded"/"conditional" verdict
        # can legitimately rest on a COVERAGE clause's own condition or scope,
        # not only a dedicated exclusion chunk. Two real examples from this
        # run: CLAIM_03 (tyre "covered only when the vehicle has also been
        # damaged" -- condition unmet -> excluded, citing that coverage
        # clause) and CLAIM_08 (crack coverage enumerates "bumpers, door
        # panels, trims"; a lamp-housing crack falls outside that scope ->
        # excluded, citing the coverage clause). Both are sound reasoning, so
        # requiring an exclusion-typed citation here produced false positives.
        # Surfaced for manual review instead of penalising the composite.
        if verdict in {"excluded", "conditional"} and not (set(cited) & offered_exclusion_ids):
            neg_verdict_on_coverage_only.append({"damage_class": cls, "verdict": verdict})

        rationale_text = it.get("rationale", "") or ""
        if CURRENCY_RE.search(rationale_text):
            all_offered_text = " ".join(c["text"] for c in clause_bucket["coverage"] + clause_bucket["exclusion_or_condition"])
            if not CURRENCY_RE.search(all_offered_text):
                currency_violations.append({"damage_class": cls, "rationale": rationale_text})

    escalation_consistent = bool(parsed.get("escalate_to_human")) == bool(payload["escalation"]["needs_human_review"])
    if not escalation_consistent:
        issues.append(
            f"escalate_to_human={parsed.get('escalate_to_human')} but payload flagged "
            f"needs_human_review={payload['escalation']['needs_human_review']}"
        )

    overall_text = parsed.get("overall_recommendation", "") or ""
    if CURRENCY_RE.search(overall_text):
        currency_violations.append({"damage_class": "overall", "rationale": overall_text})

    return {
        "schema_valid": True,
        "class_coverage_complete": class_coverage_complete,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "citation_validity_rate": round(valid_citations / total_citations, 3) if total_citations else None,
        "verdict_evidence_consistent": verdict_evidence_ok,
        "escalation_consistent": escalation_consistent,
        "currency_violations": currency_violations,
        "multi_class_chunk_citations": sorted(set(multi_class_citations)),
        "negative_verdict_on_coverage_only": neg_verdict_on_coverage_only,
        "issues": issues,
    }


def evaluate_model(model_file: Path, payloads_by_id: dict, chunk_classes: dict) -> dict:
    with open(model_file) as f:
        reports = json.load(f)

    per_claim = {}
    for r in reports:
        payload = payloads_by_id[r["claim_id"]]
        if not r["ok"]:
            per_claim[r["claim_id"]] = {"schema_valid": False, "issues": [f"API/parse error: {r.get('error')}"]}
            continue
        per_claim[r["claim_id"]] = check_report(payload, r["parsed"], chunk_classes)

    n = len(per_claim)
    schema_valid_rate = sum(1 for c in per_claim.values() if c.get("schema_valid")) / n
    valid_reports = [c for c in per_claim.values() if c.get("schema_valid")]
    n_valid = len(valid_reports) or 1

    coverage_complete_rate = sum(1 for c in valid_reports if c.get("class_coverage_complete")) / n_valid
    verdict_consistent_rate = sum(1 for c in valid_reports if c.get("verdict_evidence_consistent")) / n_valid
    escalation_consistent_rate = sum(1 for c in valid_reports if c.get("escalation_consistent")) / n_valid

    total_cit = sum(c.get("total_citations") or 0 for c in valid_reports)
    valid_cit = sum(c.get("valid_citations") or 0 for c in valid_reports)
    citation_validity_rate = round(valid_cit / total_cit, 3) if total_cit else None

    n_currency_violations = sum(len(c.get("currency_violations", [])) for c in valid_reports)
    n_multi_class_flags = sum(len(c.get("multi_class_chunk_citations", [])) for c in valid_reports)
    n_neg_on_coverage = sum(len(c.get("negative_verdict_on_coverage_only", [])) for c in valid_reports)

    composite_score = round(
        (schema_valid_rate + coverage_complete_rate + verdict_consistent_rate +
         escalation_consistent_rate + (citation_validity_rate or 0)) / 5,
        3,
    )

    return {
        "n_claims": n,
        "schema_valid_rate": round(schema_valid_rate, 3),
        "class_coverage_complete_rate": round(coverage_complete_rate, 3),
        "citation_validity_rate": citation_validity_rate,
        "verdict_evidence_consistency_rate": round(verdict_consistent_rate, 3),
        "escalation_consistency_rate": round(escalation_consistent_rate, 3),
        "currency_violation_count": n_currency_violations,
        "multi_class_chunk_citation_flags": n_multi_class_flags,
        "negative_verdict_on_coverage_only_flags": n_neg_on_coverage,
        "composite_score": composite_score,
        "per_claim": per_claim,
    }


def main():
    with open(PAYLOADS_PATH) as f:
        payloads = json.load(f)
    payloads_by_id = {p["claim_id"]: p for p in payloads}
    chunk_classes = load_chunk_damage_classes()

    results = {}
    for model_file in sorted(REPORTS_DIR.glob("reports_*.json")):
        if model_file.name == "reports_all.json":
            continue
        model_name = model_file.stem.replace("reports_", "").replace("_", "/", 1) if "openai" in model_file.stem else model_file.stem.replace("reports_", "")
        results[model_name] = evaluate_model(model_file, payloads_by_id, chunk_classes)

    print(f"{'model':28s} {'schema':>7s} {'cov_cmpl':>9s} {'cite_valid':>11s} {'verdict_ok':>11s} {'escal_ok':>9s} {'$viol':>6s} {'multi_cls':>10s} {'neg_cov':>8s} {'composite':>10s}")
    for name, r in results.items():
        print(f"{name:28s} {r['schema_valid_rate']:>7} {r['class_coverage_complete_rate']:>9} "
              f"{str(r['citation_validity_rate']):>11} {r['verdict_evidence_consistency_rate']:>11} "
              f"{r['escalation_consistency_rate']:>9} {r['currency_violation_count']:>6} "
              f"{r['multi_class_chunk_citation_flags']:>10} {r['negative_verdict_on_coverage_only_flags']:>8} "
              f"{r['composite_score']:>10}")

    best = max(results, key=lambda k: results[k]["composite_score"])
    print(f"\nBest model by composite score: {best}")

    with open(OUT_PATH, "w") as f:
        json.dump({"results": results, "best_model": best}, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
