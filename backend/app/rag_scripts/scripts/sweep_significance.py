"""
Are the sweep differences real, or are they one chunk moving?

sweep_rag_params.py reports P@3 to four decimals across a few hundred
configurations, which invites reading a 0.9200 as beating a 0.9133. At n=50
incidents that difference is one retrieved chunk in one incident's top-3:
P@3 averages 50 scores each of which is (hits in top 3)/3, so the whole
metric moves in quanta of 1/150 = 0.00667. A 0.0067 gap is one quantum.

This script puts an interval around each gap instead. For every candidate
configuration it computes the per-incident P@3 vector, pairs it against the
production configuration incident-by-incident, and bootstraps the mean
difference over incidents (10,000 resamples, fixed seed). A 95% interval
spanning zero means the configuration is not distinguishable from production
on this evaluation set -- which is the honest reading of nearly all of them.

    python scripts/sweep_significance.py
"""
import json
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.sweep_rag_params import (
    CANDIDATE_POOL,
    CHUNKS_TSV,
    DB_PATH,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    GT_PATH,
    INCIDENTS_PATH,
    RRF_K,
    SPARSE_WEIGHT,
    build_cache,
    fuse,
    load_corpus,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "rag_outputs" / "eval" / "sweep_significance.json"

N_BOOTSTRAP = 10_000
SEED = 20260807

# (label, dense_w, sparse_w, rrf_k, pool)
CANDIDATES = [
    ("production (3:1, k=60, pool=20)", DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K, CANDIDATE_POOL),
    ("dense-only (1:0)", 1.0, 0.0, RRF_K, CANDIDATE_POOL),
    ("sparse-only (0:1)", 0.0, 1.0, RRF_K, CANDIDATE_POOL),
    ("ratio 2:1", 2.0, 1.0, RRF_K, CANDIDATE_POOL),
    ("ratio 1:1", 1.0, 1.0, RRF_K, CANDIDATE_POOL),
    ("ratio 4:1", 4.0, 1.0, RRF_K, CANDIDATE_POOL),
    ("rrf_k=5", DENSE_WEIGHT, SPARSE_WEIGHT, 5, CANDIDATE_POOL),
    ("rrf_k=10", DENSE_WEIGHT, SPARSE_WEIGHT, 10, CANDIDATE_POOL),
    ("rrf_k=100", DENSE_WEIGHT, SPARSE_WEIGHT, 100, CANDIDATE_POOL),
    ("rrf_k=1000", DENSE_WEIGHT, SPARSE_WEIGHT, 1000, CANDIDATE_POOL),
    ("pool=3", DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K, 3),
    ("pool=5", DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K, 5),
    ("pool=10 (grid best)", DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K, 10),
    ("pool=50", DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K, 50),
    ("pool=100", DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K, 100),
]


def per_incident_p3(cache, gt, incidents, dense_w, sparse_w, rrf_k, pool):
    """P@3 for each incident separately -- the paired sample the bootstrap needs."""
    out = []
    for inc in incidents:
        dense_ids, sparse_ids = cache[inc["incident_id"]]
        fused = fuse(dense_ids, sparse_ids, dense_w, sparse_w, rrf_k, pool)
        top3 = [cid for cid, _ in fused[:3]]
        target = set(inc["damage_classes"])
        out.append(sum(bool(set(gt[c]["damage_classes"]) & target) for c in top3) / 3)
    return np.array(out)


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)
    with open(INCIDENTS_PATH) as f:
        incidents = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = chromadb.PersistentClient(path=str(DB_PATH)).get_collection("policy_clauses")
    chunk_ids, chunk_texts, _ = load_corpus(CHUNKS_TSV)
    queries = {inc["incident_id"]: inc["description"] for inc in incidents}
    cache = build_cache(model, collection, chunk_ids, chunk_texts, queries)

    vectors = {
        label: per_incident_p3(cache, gt, incidents, dw, sw, k, p)
        for label, dw, sw, k, p in CANDIDATES
    }
    base = vectors[CANDIDATES[0][0]]

    rng = np.random.default_rng(SEED)
    n = len(incidents)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))

    rows = []
    for label, *_ in CANDIDATES:
        vec = vectors[label]
        diff = vec - base
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "config": label,
            "mean_p3": round(float(vec.mean()), 4),
            "delta_vs_production": round(float(diff.mean()), 4),
            "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4),
            "incidents_differing": int((diff != 0).sum()),
            "top3_slots_differing": int(round(abs(diff).sum() * 3)),
            "significant": bool(lo > 0 or hi < 0),
        })

    results = {
        "method": (
            f"paired bootstrap over {n} incidents, {N_BOOTSTRAP} resamples, seed {SEED}; "
            "baseline = production config"
        ),
        "p3_quantum": round(1 / (3 * n), 5),
        "quantum_note": "smallest possible change in mean P@3 = one top-3 slot on one incident",
        "comparisons": rows,
    }

    print(f"{'config':34} {'P@3':>7} {'delta':>8} {'95% CI':>18} {'inc':>4} {'sig':>4}")
    for r in rows:
        ci = f"[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]"
        print(f"{r['config']:34} {r['mean_p3']:>7.4f} {r['delta_vs_production']:>+8.4f} "
              f"{ci:>18} {r['incidents_differing']:>4} {'YES' if r['significant'] else 'no':>4}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
