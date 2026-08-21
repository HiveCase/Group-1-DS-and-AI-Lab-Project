from __future__ import annotations

import logging
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from app.core.config import get_settings
from app.services.policy_service import POLICY_PDF_MAP

logger = logging.getLogger("claims_portal.policy_clause_service")

# backend/app/rag_scripts holds the hybrid dense+sparse clause-retrieval
# pipeline (src/retrieval/*) that used to be evaluated standalone and never
# wired into the app. It is an internal library, not a callable service, so
# it is reached the same way any sibling package would be: by putting its
# root on sys.path once. Imports of its `src.*` modules are done lazily
# inside _rag_modules() so importing this file never pays the
# chromadb/sentence-transformers/scikit-learn cost unless retrieval is
# actually used, and so the service degrades gracefully (see
# _get_clause_retriever) if those heavy deps are missing in an environment.
_RAG_ROOT = Path(__file__).resolve().parents[1] / "rag_scripts"
_SYNTHETIC_PDF_DIR = _RAG_ROOT / "data" / "policy_pdfs" / "synthetic"


def _bootstrap_rag_path() -> None:
    root_str = str(_RAG_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _rag_modules() -> dict[str, Any]:
    _bootstrap_rag_path()
    from src.retrieval.clause_retriever import ClauseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.policy_parser import tag_damage_classes
    from src.retrieval.policy_store import has_policy, ingest_policy

    return {
        "HybridRetriever": HybridRetriever,
        "ClauseRetriever": ClauseRetriever,
        "has_policy": has_policy,
        "ingest_policy": ingest_policy,
        "tag_damage_classes": tag_damage_classes,
    }


class PolicyClauseService:
    """Retrieves source-cited policy clauses for a claim's detected damage.

    Each seeded policy is ingested once (on first use, or eagerly via
    ensure_all_seeded_policies_ingested at app startup) into its own Chroma
    collection under backend/app/rag_scripts/data/claim_pipeline/policies,
    keyed by policy_number as the retrieval pipeline's "user_id". Retrieval
    itself is delegated to the existing hybrid dense+sparse ClauseRetriever
    rather than reimplemented here.
    """

    def __init__(self, retriever_factory: Callable[[str], Any] | None = None):
        self.settings = get_settings()
        self._retriever_factory = retriever_factory or self._default_retriever_factory
        self._retriever_cache: dict[str, Any] = {}

    def retrieve_clauses(
        self,
        policy_number: str | None,
        damage_classes: list[str] | None,
        incident_description: str = "",
        claimed_amount: Decimal | str | None = None,
        policy_limit: Decimal | str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        if policy_number:
            classes = [c for c in (damage_classes or []) if c and c != "unknown"]
            if not classes:
                classes = self._infer_damage_classes(incident_description)
            if classes:
                clause_retriever = self._get_clause_retriever(policy_number)
                if clause_retriever is not None:
                    findings.extend(self._collect_findings(clause_retriever, classes, top_k))

        limit_finding = self._policy_limit_finding(claimed_amount, policy_limit)
        if limit_finding is not None:
            findings.append(limit_finding)
        return findings

    def ensure_all_seeded_policies_ingested(self) -> None:
        """Ingest every seeded policy PDF up front so the first claim
        submitted against a given policy doesn't pay the embedding cost
        inline. Failures are logged and skipped per-policy so a missing or
        unparsable PDF for one policy doesn't block app startup."""
        for policy_number in POLICY_PDF_MAP:
            try:
                self._ensure_ingested(policy_number)
            except Exception:
                logger.exception("Failed to ingest policy clauses for %s", policy_number)

    def _collect_findings(self, clause_retriever: Any, classes: list[str], top_k: int) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for damage_class in classes:
            try:
                result = clause_retriever.get_clauses(damage_class)
            except ValueError:
                continue
            hits = list(result.get("coverage") or [])[:top_k] + list(result.get("exclusion_or_condition") or [])[:top_k]
            for hit in hits:
                chunk_id = hit.get("chunk_id")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                findings.append({
                    "clause_id": chunk_id,
                    "text": hit.get("text"),
                    "source_citation": hit.get("heading") or hit.get("doc_id") or "unknown",
                    "clause_type": hit.get("clause_type"),
                    "damage_class": damage_class,
                    "score": hit.get("score"),
                    # Explainability: why this clause was retrieved, not
                    # just that it was -- the dense (semantic) vs. sparse
                    # (keyword) contribution behind `score`, when available.
                    "retrieval_breakdown": hit.get("retrieval_breakdown"),
                })
        return findings

    def _policy_limit_finding(self, claimed_amount: Decimal | str | None, policy_limit: Decimal | str | None) -> dict[str, Any] | None:
        if policy_limit is None or claimed_amount is None:
            return None
        limit = Decimal(str(policy_limit))
        amount = Decimal(str(claimed_amount))
        within_limit = amount <= limit
        return {
            "clause_id": "POLICY-LIMIT-CHECK",
            "text": (
                f"Claimed amount {amount} is within the policy limit of {limit}."
                if within_limit
                else f"Claimed amount {amount} exceeds the policy limit of {limit}."
            ),
            "source_citation": "policy declarations",
            "clause_type": "sub_limit",
            "damage_class": None,
            "score": None,
            "status": "within_policy_limit" if within_limit else "outside_policy_limit",
            "policy_limit": str(limit),
            "claimed_amount": str(amount),
        }

    def _infer_damage_classes(self, incident_description: str) -> list[str]:
        if not incident_description:
            return []
        try:
            rag = _rag_modules()
        except Exception:
            logger.exception("Unable to load policy-clause retrieval modules for damage-class inference")
            return []
        return rag["tag_damage_classes"](incident_description)

    def _get_clause_retriever(self, policy_number: str) -> Any | None:
        if policy_number in self._retriever_cache:
            return self._retriever_cache[policy_number]
        try:
            retriever = self._retriever_factory(policy_number)
        except Exception:
            logger.exception("Unable to build clause retriever for policy %s", policy_number)
            return None
        self._retriever_cache[policy_number] = retriever
        return retriever

    def _default_retriever_factory(self, policy_number: str) -> Any:
        rag = _rag_modules()
        self._ensure_ingested(policy_number, rag=rag)
        hybrid = rag["HybridRetriever"](user_id=policy_number)
        return rag["ClauseRetriever"](hybrid)

    def _ensure_ingested(self, policy_number: str, rag: dict[str, Any] | None = None) -> None:
        rag = rag or _rag_modules()
        if rag["has_policy"](policy_number):
            return
        pdf_filename = POLICY_PDF_MAP.get(policy_number)
        if not pdf_filename:
            raise ValueError(f"No policy PDF mapped for policy '{policy_number}'")
        pdf_path = _SYNTHETIC_PDF_DIR / pdf_filename
        rag["ingest_policy"](pdf_path, user_id=policy_number)
