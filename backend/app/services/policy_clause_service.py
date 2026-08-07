from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from app.core.config import get_settings


class PolicyClauseService:
    def __init__(self, collection: Any | None = None, embedder: Any | None = None):
        self.collection = collection
        self.embedder = embedder
        self.settings = get_settings()

    def retrieve_clauses(self, claim_type: str, amount: Decimal | str | None, top_k: int = 3) -> list[dict[str, Any]]:
        query_text = self._build_query(claim_type, amount)
        collection = self._get_collection()
        if collection is None:
            return []
        results = collection.query(query_text, n_results=top_k)
        return [
            {
                "clause_id": item.get("id"),
                "text": item.get("text"),
                "source_citation": item.get("metadata", {}).get("source") or item.get("metadata", {}).get("section") or "unknown",
                "metadata": item.get("metadata", {}),
            }
            for item in results
        ]

    def ingest_policy_pdf(self, pdf_path: str | Path | None = None) -> list[dict[str, Any]]:
        pdf_path = Path(pdf_path or self.settings.data_dir / "policy_clause.pdf")
        if not pdf_path.exists():
            return []
        try:
            import pdfplumber
        except Exception:
            return []

        clauses: list[dict[str, Any]] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text:
                    continue
                for chunk in self._split_text(text):
                    clauses.append({
                        "id": f"CL-{page_index + 1}-{len(clauses) + 1}",
                        "text": chunk,
                        "metadata": {"source": f"page {page_index + 1}", "section": f"section {page_index + 1}"},
                    })
        return clauses

    def _split_text(self, text: str) -> list[str]:
        parts = [part.strip() for part in text.split("\n\n") if part.strip()]
        return parts[:5]

    def _build_query(self, claim_type: str, amount: Decimal | str | None) -> str:
        amount_text = str(amount or "")
        return f"{claim_type} {amount_text} coverage claim".strip()

    def _get_collection(self) -> Any | None:
        return self.collection

    def _embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]
