"""
Ingest one user-uploaded policy PDF into its own scoped RAG index.

Replaces the fixed 5-policy catalog (formerly scripts/policy_catalog.py +
scripts/policy_selector.py, both removed): a real deployment has no menu to
choose from, because coverage terms are only meaningful when they're the
policy the claimant actually holds. Each user uploads their own policy PDF,
and it is parsed, chunked, tagged, and embedded into a ChromaDB collection
named "user_{user_id}" that contains that one document and nothing else --
retrieval scoped to it (scripts/hybrid_retrieval.py's HybridRetriever.for_user)
can never surface another user's clauses, because no other user's clauses
are in the collection.

Reuses the parsing/chunking/tagging functions from
scripts/preprocess_policy_pdfs.py unchanged -- the corpus-quality lessons
baked into those functions (heading-aware chunking, word-boundary keyword
tagging, heading-priority clause-type tagging) apply just as much to a
single uploaded PDF as to the original 5-document batch job. Only the
indexing target changes: one small per-user ChromaDB collection + TSV
instead of one shared multi-policy collection.

Run as a script:
    python scripts/ingest_user_policy.py --pdf path/to/policy.pdf --user_id alice
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from hybrid_retrieval import USER_CHUNKS_DIR, USER_DB_PATH
from preprocess_policy_pdfs import (
    chunk_document,
    clean_text,
    contextualize,
    dedup_chunks,
    extract_pdf_text,
    tag_clause_type,
    tag_damage_classes,
)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
DEDUP_THRESHOLD = 0.90


def ingest(pdf_path: Path, user_id: str, embedding_model: str = "all-MiniLM-L6-v2") -> dict:
    """Parse, chunk, tag, and embed one policy PDF into the "user_{user_id}"
    ChromaDB collection, replacing any previous policy that user had
    ingested (re-uploading a corrected or renewed policy should not leave
    the old one's clauses retrievable alongside it)."""
    raw = extract_pdf_text(pdf_path)
    clean = clean_text(raw, pdf_path.name)
    chunks_raw = chunk_document(clean, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks_dedup = dedup_chunks(chunks_raw, threshold=DEDUP_THRESHOLD)

    doc_id = f"user_{user_id}"
    texts, metas = [], []
    for chunk_index, (heading, chunk) in enumerate(chunks_dedup):
        ctx_text = contextualize(heading, chunk)
        texts.append(ctx_text)
        metas.append({
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "damage_classes": tag_damage_classes(ctx_text),
            "clause_type": tag_clause_type(heading, chunk),
            "heading": heading,
        })

    model = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(USER_DB_PATH))
    collection_name = f"user_{user_id}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    ids = [f"chunk_{i:05d}" for i in range(len(texts))]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    safe_meta = [{
        "doc_id": m["doc_id"],
        "heading": m["heading"],
        "damage_classes": ",".join(m["damage_classes"]),
        "clause_type": m["clause_type"],
        "chunk_index": m["chunk_index"],
    } for m in metas]
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=safe_meta)

    USER_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = USER_CHUNKS_DIR / f"{user_id}.tsv"
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["chunk_id", "doc_id", "heading", "damage_classes", "clause_type", "chunk_len", "text"])
        for cid, text, meta in zip(ids, texts, metas):
            writer.writerow([
                cid, meta["doc_id"], meta["heading"],
                ",".join(meta["damage_classes"]), meta["clause_type"],
                len(text), text.replace("\n", " "),
            ])

    return {
        "user_id": user_id,
        "doc_id": doc_id,
        "source_pdf": pdf_path.name,
        "ingested_at": datetime.now().isoformat(),
        "n_chunks": len(texts),
        "collection": collection_name,
        "chunks_tsv": str(tsv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest one user's uploaded policy PDF into its own scoped RAG index")
    parser.add_argument("--pdf", required=True, help="Path to the uploaded policy PDF")
    parser.add_argument("--user_id", required=True, help="Unique user/session identifier")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    summary = ingest(Path(args.pdf), args.user_id, args.embedding_model)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
