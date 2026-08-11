"""
Per-user policy ingestion and lookup.

Each user's uploaded policy is embedded into its own ChromaDB collection
("user_{user_id}") holding that one document and nothing else. Isolation is
therefore structural rather than a filter applied at query time: retrieval
scoped to a user's collection cannot surface another user's clauses because
no other user's clauses are in it, so there is no filter to forget to pass
and no way for a bug elsewhere to widen the scope.

This is the persistent thing in the architecture. A user ingests once; every
later claim they file reads this index. Claims themselves are separate
short-lived graph runs (src/agents/graph.py) that never mutate it.
"""
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chromadb

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEDUP_THRESHOLD,
    EMBEDDING_MODEL,
    USER_CHUNKS_DIR,
    USER_DB_PATH,
    USER_MANIFEST_DIR,
)
from src.retrieval.policy_parser import parse_policy_pdf

log = logging.getLogger(__name__)


@dataclass
class PolicyManifest:
    """What we know about a user's ingested policy, without re-reading the
    index. Written at ingest time so `list-users` / claim validation can
    answer "has this user uploaded a policy?" cheaply."""
    user_id: str
    doc_id: str
    source_pdf: str
    n_chunks: int
    ingested_at: str
    embedding_model: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def collection_name(user_id: str) -> str:
    return f"user_{user_id}"


def manifest_path(user_id: str) -> Path:
    return USER_MANIFEST_DIR / f"{user_id}.json"


def chunks_tsv_path(user_id: str) -> Path:
    return USER_CHUNKS_DIR / f"{user_id}.tsv"


def has_policy(user_id: str) -> bool:
    return manifest_path(user_id).exists() and chunks_tsv_path(user_id).exists()


def load_manifest(user_id: str) -> PolicyManifest:
    if not has_policy(user_id):
        raise FileNotFoundError(
            f"User '{user_id}' has no ingested policy. Run: python -m src.cli ingest-policy "
            f"--user-id {user_id} --pdf <path-to-policy.pdf>"
        )
    with open(manifest_path(user_id)) as f:
        return PolicyManifest(**json.load(f))


def list_users() -> list:
    if not USER_MANIFEST_DIR.exists():
        return []
    return sorted(p.stem for p in USER_MANIFEST_DIR.glob("*.json"))


def ingest_policy(pdf_path, user_id: str, embedding_model: str = EMBEDDING_MODEL) -> PolicyManifest:
    """Parse, chunk, tag, embed one user's policy PDF into their own
    collection, replacing whatever they had before.

    The replace is deliberate and total (delete_collection, then rewrite the
    TSV): a user re-uploading a renewed or corrected policy must not leave
    the superseded version's clauses retrievable alongside the new one, or a
    claim could be assessed against terms that no longer apply.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Policy PDF not found: {pdf_path}")

    chunks = parse_policy_pdf(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP, DEDUP_THRESHOLD)
    if not chunks:
        raise ValueError(
            f"No usable text extracted from {pdf_path.name}. If it is a scanned "
            "image PDF, it needs OCR before ingestion -- this pipeline does not do OCR."
        )

    # Imported lazily -- loading the sentence-transformer costs seconds, and
    # callers that only read manifests shouldn't pay for it.
    from sentence_transformers import SentenceTransformer

    doc_id = collection_name(user_id)
    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{c['chunk_index']:05d}" for c in chunks]

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    USER_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(USER_DB_PATH))
    try:
        client.delete_collection(doc_id)
    except Exception:
        pass  # first upload for this user
    collection = client.get_or_create_collection(doc_id, metadata={"hnsw:space": "cosine"})
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{
            "doc_id": doc_id,
            "heading": c["heading"],
            "damage_classes": ",".join(c["damage_classes"]),
            "clause_type": c["clause_type"],
            "chunk_index": c["chunk_index"],
        } for c in chunks],
    )

    USER_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(chunks_tsv_path(user_id), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["chunk_id", "doc_id", "heading", "damage_classes",
                         "clause_type", "chunk_len", "text"])
        for cid, c in zip(ids, chunks):
            writer.writerow([
                cid, doc_id, c["heading"], ",".join(c["damage_classes"]),
                c["clause_type"], len(c["text"]), c["text"].replace("\n", " "),
            ])

    manifest = PolicyManifest(
        user_id=user_id,
        doc_id=doc_id,
        source_pdf=pdf_path.name,
        n_chunks=len(chunks),
        ingested_at=datetime.now().isoformat(timespec="seconds"),
        embedding_model=embedding_model,
    )
    USER_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path(user_id), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    log.info("Ingested %s for user '%s': %d chunks", pdf_path.name, user_id, len(chunks))
    return manifest
