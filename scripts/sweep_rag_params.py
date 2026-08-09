"""
Hyperparameter sweeps for the retrieval stack.

Milestone 2 swept exactly one retrieval hyperparameter -- the dense:sparse
RRF weight ratio -- and settled the rest by inspection or by leaving library
defaults in place. This script closes that gap: it sweeps every remaining
retrieval knob against the same 50-incident evaluation the weight ratio was
tuned on, so each value in src/config.py can be defended with a number
instead of an assertion.

Knobs swept here
    A  dense:sparse weight ratio     (reproduction of the published sweep)
    B  RRF_K                         (rank-fusion damping constant)
    C  CANDIDATE_POOL                (per-signal candidate depth)
    D  RRF_K x CANDIDATE_POOL        (interaction, since B and C are coupled)
    E  MIN_CLAUSE_SCORE              (fused-score noise floor)
    F  CHUNK_SIZE / CHUNK_OVERLAP / DEDUP_THRESHOLD

Method. Sweeps A-E never change the corpus, so the dense (ChromaDB) and
sparse (TF-IDF) rankings depend only on the query, not on the fusion
parameters. Both rankings are therefore computed once per incident at
maximum depth and cached; every (weight, RRF_K, pool) combination is then
pure arithmetic over those cached rankings. This is what makes a 3-way grid
cheap enough to run exhaustively rather than sampling it.

Sweep F does change the corpus, so it re-chunks, re-tags, re-embeds and
rebuilds a throwaway ChromaDB collection per configuration.

Relevance judgements. `data/clause_groundtruth.json` is not hand-labelled:
its damage_classes are the same regex auto-tags that policy_parser.py writes
onto each chunk (verified identical for all 185 chunks). A retrieved chunk
counts as relevant when its auto-tags intersect the incident's damage
classes. That is what makes sweep F legitimate -- re-chunking regenerates
the labels by the same rule rather than invalidating a fixed answer key --
but it also means these scores measure retrieval against the tagger, not
against a human adjudicator. Section F reports a random-retrieval baseline
alongside each configuration for exactly this reason: chunkings differ in
how dense relevant chunks are, and raw P@3 is not comparable across them
without it.

Usage
    python scripts/sweep_rag_params.py                 # sweeps A-E (seconds)
    python scripts/sweep_rag_params.py --with-chunking # adds F (minutes)
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.policy_parser import (
    chunk_document,
    clean_text,
    contextualize,
    dedup_chunks,
    extract_pdf_text,
    tag_clause_type,
    tag_damage_classes,
)

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "data" / "clause_groundtruth.json"
CHUNKS_TSV = ROOT / "data" / "rag_outputs" / "chunks_all.tsv"
INCIDENTS_PATH = ROOT / "data" / "rag_outputs" / "eval" / "incident_descriptions.json"
DB_PATH = ROOT / "data" / "chroma_db"
PDF_DIR = ROOT / "data" / "policy_pdfs" / "synthetic"
OUT_PATH = ROOT / "data" / "rag_outputs" / "eval" / "hyperparameter_sweeps.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Production values (src/config.py), held fixed while another axis is swept.
DENSE_WEIGHT = 3.0
SPARSE_WEIGHT = 1.0
RRF_K = 60
CANDIDATE_POOL = 20
RETRIEVAL_POOL = 15
CLAUSES_PER_TYPE = 5
MIN_CLAUSE_SCORE = 0.01
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
DEDUP_THRESHOLD = 0.90
COVERAGE_CLAUSE_TYPES = {"coverage", "definition"}
EXCLUSION_CLAUSE_TYPES = {"exclusion", "sub_limit", "condition"}

# Deepest ranking any swept configuration can ask for; cache to this depth.
MAX_POOL = 100

# Clause queries, copied from src/retrieval/clause_retriever.py so sweep E
# measures the floor against the traffic that actually meets it.
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


# ---------------------------------------------------------------- fusion core

def fuse(dense_ids, sparse_ids, dense_w, sparse_w, rrf_k, pool):
    """Weighted Reciprocal Rank Fusion over two ranked id lists.

    Mirrors HybridRetriever._fused_ranking exactly, including the detail that
    a zero-weighted signal still inserts its ids at score 0.0 rather than
    being skipped -- dict insertion order is the tie-break under Python's
    stable sort, so skipping them would silently reorder ties and stop this
    reproducing the published dense-only numbers.
    """
    scores = {}
    for rank, cid in enumerate(dense_ids[:pool], 1):
        scores[cid] = scores.get(cid, 0.0) + dense_w / (rrf_k + rank)
    for rank, cid in enumerate(sparse_ids[:pool], 1):
        scores[cid] = scores.get(cid, 0.0) + sparse_w / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def evaluate(cache, gt, incidents, dense_w, sparse_w, rrf_k, pool):
    """Mean P@3 / MRR@5 over the incident set, as scripts/hybrid_retrieval.py
    computes them: P@3 always divides by 3, and MRR only looks down to rank 5."""
    p3_scores, rr_scores, zero_hit = [], [], []

    for inc in incidents:
        dense_ids, sparse_ids = cache[inc["incident_id"]]
        fused = fuse(dense_ids, sparse_ids, dense_w, sparse_w, rrf_k, pool)
        top5 = [cid for cid, _ in fused[:5]]
        target = set(inc["damage_classes"])

        hits3 = [bool(set(gt[cid]["damage_classes"]) & target) for cid in top5[:3]]
        p3 = sum(hits3) / 3
        p3_scores.append(p3)
        if p3 == 0:
            zero_hit.append(inc["incident_id"])

        rr = 0.0
        for rank, cid in enumerate(top5, 1):
            if set(gt[cid]["damage_classes"]) & target:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return {
        "mean_precision_at_3": round(sum(p3_scores) / len(p3_scores), 4),
        "mean_reciprocal_rank": round(sum(rr_scores) / len(rr_scores), 4),
        "zero_hit_count": len(zero_hit),
        "zero_hit_incidents": zero_hit,
    }


def random_baseline(gt, incidents):
    """Expected P@3 from retrieving uniformly at random: the mean share of the
    corpus that is relevant to an incident. Sweep F changes corpus size and
    tag density, so raw P@3 there is only interpretable against this."""
    all_ids = list(gt)
    densities = []
    for inc in incidents:
        target = set(inc["damage_classes"])
        rel = sum(1 for cid in all_ids if set(gt[cid]["damage_classes"]) & target)
        densities.append(rel / len(all_ids))
    return round(sum(densities) / len(densities), 4)


# ------------------------------------------------------------------- corpora

def load_corpus(chunks_tsv):
    """chunk_id -> metadata, preserving TSV order for the TF-IDF row mapping."""
    ids, texts, meta = [], [], {}
    with open(chunks_tsv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids.append(row["chunk_id"])
            texts.append(row["text"])
            meta[row["chunk_id"]] = {
                "doc_id": row["doc_id"],
                "heading": row["heading"],
                "clause_type": row["clause_type"],
                "damage_classes": [c for c in row["damage_classes"].split(",") if c],
                "text": row["text"],
            }
    return ids, texts, meta


def build_cache(model, collection, chunk_ids, chunk_texts, queries):
    """Dense and sparse rankings at MAX_POOL depth for every query.

    Computed once because no fusion parameter can change either ranking --
    only how the two are combined. Truncating a depth-100 ranking to depth p
    is identical to having asked for depth p.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(chunk_texts)

    keys = list(queries)
    embeddings = model.encode([queries[k] for k in keys]).tolist()
    depth = min(MAX_POOL, len(chunk_ids))

    cache = {}
    for key, emb in zip(keys, embeddings):
        dense_ids = collection.query(
            query_embeddings=[emb], n_results=depth, include=[]
        )["ids"][0]
        sims = cosine_similarity(vectorizer.transform([queries[key]]), tfidf)[0]
        sparse_ids = [chunk_ids[i] for i in sims.argsort()[::-1][:depth]]
        cache[key] = (dense_ids, sparse_ids)
    return cache


def rebuild_corpus(model, client, tag, chunk_size, chunk_overlap, dedup_threshold):
    """Re-chunk, re-tag and re-embed the 5 synthetic policies into a throwaway
    collection. Returns the same (ids, texts, meta, gt, collection) shape the
    prebuilt corpus provides, with groundtruth regenerated by the same tagger."""
    ids, texts, meta, gt = [], [], {}, {}
    n = 0
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        doc_id = pdf_path.stem
        text = clean_text(extract_pdf_text(pdf_path))
        chunks = dedup_chunks(
            chunk_document(text, chunk_size, chunk_overlap), dedup_threshold
        )
        for heading, body in chunks:
            ctx = contextualize(heading, body)
            cid = f"chunk_{n:05d}"
            n += 1
            damage_classes = tag_damage_classes(ctx)
            record = {
                "doc_id": doc_id,
                "heading": heading,
                "clause_type": tag_clause_type(heading, body),
                "damage_classes": damage_classes,
                "text": ctx,
            }
            ids.append(cid)
            texts.append(ctx)
            meta[cid] = record
            gt[cid] = {"damage_classes": damage_classes}

    name = f"sweep_{tag}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, embeddings=model.encode(texts).tolist(), documents=texts)
    return ids, texts, meta, gt, collection


# -------------------------------------------------------------------- sweeps

def sweep_weights(cache, gt, incidents):
    """A. Reproduce the published ratio sweep as a harness self-check."""
    rows = []
    for label, dw, sw in [
        ("100:0 (dense-only)", 1.0, 0.0),
        ("50:50 (1:1)", 1.0, 1.0),
        ("66:33 (2:1)", 2.0, 1.0),
        ("75:25 (3:1)", 3.0, 1.0),
        ("80:20 (4:1)", 4.0, 1.0),
        ("83:17 (5:1)", 5.0, 1.0),
        ("0:100 (sparse-only)", 0.0, 1.0),
    ]:
        r = evaluate(cache, gt, incidents, dw, sw, RRF_K, CANDIDATE_POOL)
        rows.append({"ratio": label, "dense_weight": dw, "sparse_weight": sw, **r})
    return rows


def sweep_rrf_k(cache, gt, incidents):
    """B. RRF_K damping, at production weights and pool."""
    rows = []
    for k in [1, 2, 5, 10, 20, 30, 45, 60, 80, 100, 200, 500, 1000]:
        r = evaluate(cache, gt, incidents, DENSE_WEIGHT, SPARSE_WEIGHT, k, CANDIDATE_POOL)
        rows.append({"rrf_k": k, **r})
    return rows


def sweep_pool(cache, gt, incidents):
    """C. Candidate depth per signal, at production weights and RRF_K."""
    rows = []
    for pool in [3, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        r = evaluate(cache, gt, incidents, DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K, pool)
        rows.append({"candidate_pool": pool, **r})
    return rows


def sweep_interaction(cache, gt, incidents):
    """D. RRF_K x pool. Both shape the same score curve, so a 1-D sweep of
    either could sit at a local optimum that only holds at the other's
    current value."""
    rows = []
    for k in [10, 30, 60, 100, 200]:
        for pool in [5, 10, 20, 30, 50]:
            r = evaluate(cache, gt, incidents, DENSE_WEIGHT, SPARSE_WEIGHT, k, pool)
            rows.append({"rrf_k": k, "candidate_pool": pool,
                         "mean_precision_at_3": r["mean_precision_at_3"],
                         "mean_reciprocal_rank": r["mean_reciprocal_rank"],
                         "zero_hit_count": r["zero_hit_count"]})
    return rows


def sweep_min_clause_score(model, collection, chunk_ids, chunk_texts, meta):
    """E. The fused-score floor, measured against real clause-retrieval traffic.

    P@3 on incident descriptions cannot see this parameter -- the floor is
    applied in ClauseRetriever._filter, downstream of a different query set.
    So this replays all 12 clause queries, records the fused scores that
    actually reach the filter, and reports both the analytic lower bound on a
    fused score and the empirical minimum, then counts what each candidate
    threshold would remove.
    """
    queries = {f"cov_{c}": q for c, q in COVERAGE_QUERIES.items()}
    queries.update({f"exc_{c}": q for c, q in EXCLUSION_QUERIES.items()})
    cache = build_cache(model, collection, chunk_ids, chunk_texts, queries)

    # Lowest score any surviving candidate can carry: the weaker signal alone,
    # at the deepest rank the pool admits.
    analytic_floor = min(DENSE_WEIGHT, SPARSE_WEIGHT) / (RRF_K + CANDIDATE_POOL)

    observed = []          # every score entering the clause-type filter
    kept_scores = []       # scores of clauses the filter actually returns
    for key, (dense_ids, sparse_ids) in cache.items():
        fused = fuse(dense_ids, sparse_ids, DENSE_WEIGHT, SPARSE_WEIGHT,
                     RRF_K, CANDIDATE_POOL)[:RETRIEVAL_POOL]
        allowed = COVERAGE_CLAUSE_TYPES if key.startswith("cov_") else EXCLUSION_CLAUSE_TYPES
        kept = 0
        for cid, score in fused:
            observed.append(score)
            if meta[cid]["clause_type"] in allowed:
                if kept < CLAUSES_PER_TYPE:
                    kept_scores.append(score)
                    kept += 1

    rows = []
    for thr in [0.0, 0.005, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]:
        rows.append({
            "min_clause_score": thr,
            "pool_scores_below": sum(1 for s in observed if s < thr),
            "pool_scores_total": len(observed),
            "returned_clauses_removed": sum(1 for s in kept_scores if s < thr),
            "returned_clauses_total": len(kept_scores),
        })

    return {
        "analytic_min_possible_fused_score": round(analytic_floor, 6),
        "analytic_note": (
            "min(dense_w, sparse_w) / (RRF_K + CANDIDATE_POOL) = "
            f"{min(DENSE_WEIGHT, SPARSE_WEIGHT)} / ({RRF_K} + {CANDIDATE_POOL})"
        ),
        "observed_min_fused_score": round(min(observed), 6),
        "observed_min_returned_score": round(min(kept_scores), 6),
        "n_clause_queries": len(queries),
        "thresholds": rows,
    }


def sweep_chunking(model, incidents, scratch_dir):
    """F. Corpus-level parameters. Each configuration is a fresh corpus, so
    groundtruth, corpus size and relevant-chunk density all move with it."""
    client = chromadb.PersistentClient(path=str(scratch_dir))
    queries = {inc["incident_id"]: inc["description"] for inc in incidents}

    configs = []
    for size in [150, 200, 250, 300, 400, 500, 700, 1000]:
        configs.append((size, CHUNK_OVERLAP, DEDUP_THRESHOLD))
    for overlap in [0, 20, 60, 90, 120]:
        configs.append((CHUNK_SIZE, overlap, DEDUP_THRESHOLD))
    for dedup in [0.80, 0.85, 0.95, 1.00]:
        configs.append((CHUNK_SIZE, CHUNK_OVERLAP, dedup))

    seen, rows = set(), []
    for size, overlap, dedup in configs:
        if (size, overlap, dedup) in seen:
            continue
        seen.add((size, overlap, dedup))

        tag = f"s{size}_o{overlap}_d{int(dedup * 100)}"
        t0 = time.time()
        ids, texts, meta, gt, collection = rebuild_corpus(
            model, client, tag, size, overlap, dedup)
        cache = build_cache(model, collection, ids, texts, queries)
        r = evaluate(cache, gt, incidents, DENSE_WEIGHT, SPARSE_WEIGHT,
                     RRF_K, CANDIDATE_POOL)
        base = random_baseline(gt, incidents)

        rows.append({
            "chunk_size": size,
            "chunk_overlap": overlap,
            "dedup_threshold": dedup,
            "n_chunks": len(ids),
            "mean_chunk_chars": round(sum(len(t) for t in texts) / len(texts), 1),
            "random_baseline_p3": base,
            "mean_precision_at_3": r["mean_precision_at_3"],
            "mean_reciprocal_rank": r["mean_reciprocal_rank"],
            "zero_hit_count": r["zero_hit_count"],
            "lift_over_random": round(r["mean_precision_at_3"] / base, 2) if base else None,
            "build_seconds": round(time.time() - t0, 1),
        })
        print(f"  {tag}: {len(ids)} chunks  P@3={r['mean_precision_at_3']:.4f}  "
              f"MRR={r['mean_reciprocal_rank']:.4f}  lift={rows[-1]['lift_over_random']}")

        try:
            client.delete_collection(f"sweep_{tag}")
        except Exception:
            pass

    return rows


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-chunking", action="store_true",
                        help="also run sweep F (re-embeds the corpus per config)")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    with open(GT_PATH) as f:
        gt = json.load(f)
    with open(INCIDENTS_PATH) as f:
        incidents = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("policy_clauses")
    chunk_ids, chunk_texts, meta = load_corpus(CHUNKS_TSV)

    print(f"corpus: {len(chunk_ids)} chunks | incidents: {len(incidents)}")
    queries = {inc["incident_id"]: inc["description"] for inc in incidents}
    cache = build_cache(model, collection, chunk_ids, chunk_texts, queries)

    results = {
        "meta": {
            "n_incidents": len(incidents),
            "n_chunks": len(chunk_ids),
            "embedding_model": EMBEDDING_MODEL,
            "random_baseline_p3": random_baseline(gt, incidents),
            "production_config": {
                "DENSE_WEIGHT": DENSE_WEIGHT, "SPARSE_WEIGHT": SPARSE_WEIGHT,
                "RRF_K": RRF_K, "CANDIDATE_POOL": CANDIDATE_POOL,
                "MIN_CLAUSE_SCORE": MIN_CLAUSE_SCORE, "CHUNK_SIZE": CHUNK_SIZE,
                "CHUNK_OVERLAP": CHUNK_OVERLAP, "DEDUP_THRESHOLD": DEDUP_THRESHOLD,
            },
        },
        "A_weight_ratio": sweep_weights(cache, gt, incidents),
        "B_rrf_k": sweep_rrf_k(cache, gt, incidents),
        "C_candidate_pool": sweep_pool(cache, gt, incidents),
        "D_rrf_k_x_pool": sweep_interaction(cache, gt, incidents),
        "E_min_clause_score": sweep_min_clause_score(
            model, collection, chunk_ids, chunk_texts, meta),
    }
    print("sweeps A-E complete")

    if args.with_chunking:
        scratch = ROOT / ".sweep_chroma_tmp"
        if scratch.exists():
            shutil.rmtree(scratch)
        print("sweep F (re-embedding per config):")
        try:
            results["F_chunking"] = sweep_chunking(model, incidents, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
