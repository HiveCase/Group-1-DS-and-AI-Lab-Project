"""
Turn detected damage classes into the clause evidence the LLM reasons over.

Two queries per damage class, not one. A single "is X covered?" query
reliably ranks the coverage clause highest and buries the exclusion that
actually caps or voids it -- and a report that only ever sees the coverage
clause will confidently approve claims it shouldn't, which is the specific
failure mode this domain cannot tolerate. So each class gets a
coverage-phrased query and an exclusion-phrased query, each post-filtered to
its intended clause_type, and both buckets go to the model.

The result deliberately includes coverage_clause_found: when retrieval finds
no coverage clause for a detected damage class, that is an escalation signal
(the policy may genuinely not address this damage, or retrieval may have
failed) -- never grounds for the model to conclude "not covered" on its own.
"""
import logging

from src.config import (
    CLAUSES_PER_TYPE,
    COVERAGE_CLAUSE_TYPES,
    EXCLUSION_CLAUSE_TYPES,
    MIN_CLAUSE_SCORE,
    RETRIEVAL_POOL,
)
from src.retrieval.hybrid_retriever import HybridRetriever

log = logging.getLogger(__name__)

COVERAGE_QUERIES = {
    "dent": "Is dent damage covered under accidental external means?",
    "scratch": "Does the policy cover scratches and surface paint damage?",
    "crack": "Is cracking of bumper or body panels covered?",
    "broken_lamp": "Are broken headlamps and tail lights covered?",
    "shattered_glass": "Is windscreen and window glass damage covered?",
    "flat_tyre": "Is a flat tyre or tyre blowout covered under the policy?",
}

EXCLUSION_QUERIES = {
    "dent": "What exclusions, conditions, or limits apply to a dent claim?",
    "scratch": "What exclusions, conditions, or limits apply to a scratch claim?",
    "crack": "What exclusions, conditions, or limits apply to a crack claim?",
    "broken_lamp": "What exclusions, conditions, or limits apply to a broken lamp claim?",
    "shattered_glass": "What exclusions, conditions, or limits apply to a glass damage claim?",
    "flat_tyre": "What exclusions, conditions, or limits apply to a tyre damage claim?",
}


def _hit(chunk_id: str, score: float, meta: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": meta["text"],
        "heading": meta["heading"],
        "clause_type": meta["clause_type"],
        "doc_id": meta["doc_id"],
        "score": round(score, 4),
    }


class ClauseRetriever:
    """Wraps a user-scoped HybridRetriever with the coverage/exclusion split."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def get_clauses(self, damage_class: str) -> dict:
        if damage_class not in COVERAGE_QUERIES:
            raise ValueError(f"No clause queries defined for damage class '{damage_class}'")

        coverage_pool = self.retriever.retrieve_scored(
            COVERAGE_QUERIES[damage_class], top_k=RETRIEVAL_POOL)
        exclusion_pool = self.retriever.retrieve_scored(
            EXCLUSION_QUERIES[damage_class], top_k=RETRIEVAL_POOL)

        coverage_hits = self._filter(coverage_pool, COVERAGE_CLAUSE_TYPES)
        exclusion_hits = self._filter(exclusion_pool, EXCLUSION_CLAUSE_TYPES)

        return {
            "coverage": coverage_hits,
            "exclusion_or_condition": exclusion_hits,
            "coverage_clause_found": len(coverage_hits) > 0,
        }

    def _filter(self, pool: list, allowed_types: set) -> list:
        """Keep the top CLAUSES_PER_TYPE hits whose clause_type is in the
        target bucket and whose fused score clears the noise floor.

        On COVERAGE_CLAUSE_TYPES including "definition": this is a narrow,
        verified mitigation, not a blanket relaxation. In the original
        corpus exactly one chunk of 185 was tagged "definition" -- the
        umbrella "accidental external means" coverage grant that
        dent/scratch/crack/broken_lamp all rest on. It was mistagged because
        the auto-tagger's bare `\\bmeans\\b` pattern matched "external
        means", a manner-of-occurrence phrase rather than a defining clause.
        Excluding it was confirmed end-to-end to make the Report Agent fall
        back to an irrelevant substitute chunk and, for one model, reject a
        claim that should have been covered. The root-cause fix is to
        correct the tagger's `means` pattern; until that is done and the
        corpus re-tagged, admitting "definition" into the coverage bucket
        keeps the grant reachable.
        """
        hits = []
        for cid, score in pool:
            meta = self.retriever.chunk_meta[cid]
            if meta["clause_type"] in allowed_types and score >= MIN_CLAUSE_SCORE:
                hits.append(_hit(cid, score, meta))
            if len(hits) >= CLAUSES_PER_TYPE:
                break
        return hits


def build_context_bundle(claim_id: str, user_id: str, incident_narrative: str,
                         detections: list, clauses_by_class: dict,
                         escalation: dict) -> dict:
    """Assemble the exact JSON payload handed to the LLM.

    There is no policy-selection step and no doc_id argument: the applicable
    policy is whichever one this user uploaded. Under the earlier fixed
    5-policy catalog the claimant had to state which policy applied, because
    a 315-case census showed damage profile alone could not identify it
    (top-1 accuracy 0.20). Per-user upload dissolves that problem rather
    than solving it -- there is only ever one candidate.
    """
    return {
        "claim_id": claim_id,
        "incident_narrative": incident_narrative,
        "detections": detections,
        "policy": {
            "doc_id": f"user_{user_id}",
            "user_id": user_id,
            "selection_method": "user_uploaded",
            "clauses": clauses_by_class,
        },
        "escalation": escalation,
    }
