"""
Stage 2 of the Report Agent pipeline + the context bundle assembler.

Stage 2 (ClauseRetriever): once policy_selector.PolicySelector has picked a
single doc_id for a claim, retrieve the clauses that actually go in front of
the LLM -- separately for coverage vs. exclusion/condition/sub_limit, per
detected damage class, scoped to that one document. This is a deliberate
two-query-per-class design (a coverage-phrased query and an exclusion-phrased
query, each filtered to its intended clause_type after retrieval) rather than
one mixed top-k query: a single "is X covered?" query tends to rank the
coverage clause highest and miss the exclusion that actually caps or voids
it (this is exactly the failure mode Milestone 2 documented for retrieval in
general -- a report that only ever sees the coverage clause will confidently
approve claims it shouldn't).

ContextBundleBuilder assembles the final JSON payload that goes to the LLM:
incident narrative, YOLO detections + severity, the claimant-SELECTED policy
(with its catalog description, so the report can state which cover it is
reasoning over), and per-damage-class coverage/exclusion clauses. The policy
is an explicit input, NOT inferred from the damage -- reading all 5 policies
confirmed the damage profile cannot identify the policy (all 5 cover all 6
damage classes; see docs/rag_support_mile3.md Section 5 and the rejected
auto-selection alternative in policy_selector.py). See that doc for the full
schema discussion and rationale.
"""
from pathlib import Path

from hybrid_retrieval import HybridRetriever
from policy_selector import CLASS_QUERIES
from policy_catalog import PolicyCatalog
from yolo_schema import Detection, ESCALATION_CONFIDENCE_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent

EXCLUSION_QUERIES = {
    "dent": "What exclusions, conditions, or limits apply to a dent claim?",
    "scratch": "What exclusions, conditions, or limits apply to a scratch claim?",
    "crack": "What exclusions, conditions, or limits apply to a crack claim?",
    "broken_lamp": "What exclusions, conditions, or limits apply to a broken lamp claim?",
    "shattered_glass": "What exclusions, conditions, or limits apply to a glass damage claim?",
    "flat_tyre": "What exclusions, conditions, or limits apply to a tyre damage claim?",
}

RETRIEVAL_POOL = 15
CLAUSES_PER_TYPE = 5  # top-5 clauses per bucket, scoped to the selected doc
# Fused RRF score threshold below which a "hit" is treated as noise rather
# than a real match -- see docs/rag_support_mile3.md for how this was picked.
MIN_CLAUSE_SCORE = 0.01


def _hit_dict(chunk_id: str, score: float, meta: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": meta["text"],
        "heading": meta["heading"],
        "clause_type": meta["clause_type"],
        "doc_id": meta["doc_id"],
        "score": round(score, 4),
    }


class ClauseRetriever:
    def __init__(self, retriever: HybridRetriever = None):
        self.retriever = retriever or HybridRetriever()

    def get_clauses(self, damage_class: str, doc_id: str) -> dict:
        coverage_pool = self.retriever.retrieve_scored(
            CLASS_QUERIES[damage_class], top_k=RETRIEVAL_POOL, doc_filter=doc_id
        )
        exclusion_pool = self.retriever.retrieve_scored(
            EXCLUSION_QUERIES[damage_class], top_k=RETRIEVAL_POOL, doc_filter=doc_id
        )

        coverage_hits, exclusion_hits = [], []
        for cid, score in coverage_pool:
            meta = self.retriever.chunk_meta[cid]
            # "definition" is included here for one specific, verified reason, not as a
            # blanket relaxation: chunk_00004 (policy_1's "accidental external means"
            # clause -- the umbrella coverage grant for dent/scratch/crack/broken_lamp)
            # is the ONLY chunk in the entire 185-chunk corpus tagged "definition"
            # (processing_summary.json clause_type_distribution). It was mistagged by
            # the auto-tagger's bare `\bmeans\b` keyword (scripts/preprocess_policy_pdfs.py
            # CLAUSE_TYPE_KEYWORDS["definition"]) matching "external means" -- a
            # manner-of-occurrence phrase, not a defining clause. Confirmed end-to-end:
            # excluding it here caused the Report Agent to fall back to an irrelevant
            # substitute chunk and, for one model, an incorrect claim rejection (see
            # docs/rag_support_mile3.md, CLAIM_09). Fixing the tag itself would mean
            # rewriting Milestone 2's already-[Completed] corpus artifacts, so this is
            # a narrow retrieval-layer mitigation instead -- see that doc for the
            # root-cause fix left as a flagged recommendation.
            if meta["clause_type"] in {"coverage", "definition"} and score >= MIN_CLAUSE_SCORE:
                coverage_hits.append(_hit_dict(cid, score, meta))
            if len(coverage_hits) >= CLAUSES_PER_TYPE:
                break
        for cid, score in exclusion_pool:
            meta = self.retriever.chunk_meta[cid]
            if meta["clause_type"] in {"exclusion", "sub_limit", "condition"} and score >= MIN_CLAUSE_SCORE:
                exclusion_hits.append(_hit_dict(cid, score, meta))
            if len(exclusion_hits) >= CLAUSES_PER_TYPE:
                break

        return {
            "coverage": coverage_hits,
            "exclusion_or_condition": exclusion_hits,
            "coverage_clause_found": len(coverage_hits) > 0,
        }


class ContextBundleBuilder:
    def __init__(self, retriever: HybridRetriever = None):
        self.retriever = retriever or HybridRetriever()
        self.clause_retriever = ClauseRetriever(self.retriever)

    def build(self, claim_id: str, incident_narrative: str, detections: list,
              selected_doc_id: str) -> dict:
        """Assemble the LLM payload for one claim. selected_doc_id is the
        policy the CLAIMANT chose (Stage 1 = explicit selection via
        PolicyCatalog, not inference) -- retrieval below is scoped to it."""
        if not PolicyCatalog.is_valid(selected_doc_id):
            raise ValueError(f"selected_doc_id '{selected_doc_id}' is not a known policy")

        damage_classes = sorted({d.class_name for d in detections})

        clauses_by_class = {}
        missing_coverage = []
        for cls in damage_classes:
            clauses_by_class[cls] = self.clause_retriever.get_clauses(cls, selected_doc_id)
            if not clauses_by_class[cls]["coverage_clause_found"]:
                missing_coverage.append(cls)

        low_confidence_detections = [
            d.class_name for d in detections if d.confidence < ESCALATION_CONFIDENCE_THRESHOLD
        ]

        catalog = PolicyCatalog.describe(selected_doc_id)
        return {
            "claim_id": claim_id,
            "incident_narrative": incident_narrative,
            "detections": [d.to_dict() for d in detections],
            "policy": {
                "doc_id": selected_doc_id,
                "selection_method": "claimant_selected",
                "insurer": catalog["insurer"],
                "product": catalog["product"],
                "description": catalog["summary"],
                "clauses": clauses_by_class,
            },
            "escalation": {
                "low_confidence_detections": low_confidence_detections,
                "missing_coverage_clause_for": missing_coverage,
                "needs_human_review": bool(low_confidence_detections or missing_coverage),
            },
        }
