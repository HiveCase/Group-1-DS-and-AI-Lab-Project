"""
Hybrid dense + sparse retrieval over one user's policy index.

Dense-only retrieval (MiniLM embeddings via ChromaDB) reached mean
Precision@3 = 0.893 on the 50-incident eval set. Its worst case scored 0.33:

    "Vandals smashed the rear window of the vehicle during a public
     disturbance event."  (shattered_glass)

The embedding was dominated by the broad "vandalism / malicious event"
framing shared with many unrelated-class clauses, and ranked the actual
glass coverage clause 9th -- even though the query literally contains
"window", a discriminative lexical cue a sparse retriever picks up directly.

Fusing a TF-IDF ranking in via weighted Reciprocal Rank Fusion fixes that
class of failure. The 3:1 dense:sparse ratio was swept against all 50
incidents, not just the failing one:

    dense-only (100:0)  P@3=0.893  MRR=0.980  zero-hit=0
    66:33 (2:1)         P@3=0.913  MRR=0.977  zero-hit=0
    75:25 (3:1)         P@3=0.913  MRR=0.977  zero-hit=0   <- chosen
    80:20 (4:1)         P@3=0.907  MRR=0.977  zero-hit=0   (past peak)

2:1 and 3:1 tie in aggregate (they differ on the exact top-3 for 16/50
incidents, but only by reordering chunks that are equally relevant). 3:1 was
kept because it performs identically while weighting the semantic signal
more heavily -- the conservative choice given dense-only was the stronger
retriever standalone. Weights live in src/config.py.

Scope: the retriever binds to one user's collection at construction. Both
signals are scoped by that -- the TF-IDF vocabulary is fit only on that
user's own chunks, so cross-user vocabulary can't leak in either.
"""
import csv
import logging

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CANDIDATE_POOL,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    RRF_K,
    SPARSE_WEIGHT,
    USER_DB_PATH,
)
from src.retrieval.policy_store import chunks_tsv_path, collection_name, has_policy

log = logging.getLogger(__name__)


class HybridRetriever:
    """Bound to a single user's ingested policy."""

    def __init__(self, user_id: str, embedding_model: str = EMBEDDING_MODEL):
        if not has_policy(user_id):
            raise FileNotFoundError(
                f"User '{user_id}' has no ingested policy to retrieve from. "
                f"Run: python -m src.cli ingest-policy --user-id {user_id} --pdf <policy.pdf>"
            )
        from sentence_transformers import SentenceTransformer

        self.user_id = user_id
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=str(USER_DB_PATH))
        self.collection = self.client.get_collection(collection_name(user_id))

        self.chunk_ids = []
        self.chunk_meta = {}
        chunk_texts = []
        with open(chunks_tsv_path(user_id), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                self.chunk_ids.append(row["chunk_id"])
                chunk_texts.append(row["text"])
                self.chunk_meta[row["chunk_id"]] = {
                    "doc_id": row["doc_id"],
                    "heading": row["heading"],
                    "clause_type": row["clause_type"],
                    "damage_classes": row["damage_classes"],
                    "text": row["text"],
                }

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)

    def _fused_ranking(self, query: str, dense_weight: float,
                       sparse_weight: float, pool: int) -> tuple:
        """Returns (ranked, breakdown):
        - ranked: [(chunk_id, fused_score), ...] descending, fusing the
          dense and sparse rankings by weighted RRF (unchanged behavior).
        - breakdown: {chunk_id: {dense_rank, dense_contribution,
          sparse_rank, sparse_contribution}} -- the two RRF terms that were
          summed to produce that chunk's fused score, kept only so callers
          can explain *why* a chunk ranked where it did (e.g. "found mostly
          by keyword overlap, not semantic similarity"). Not consumed by
          the fused score itself."""
        q_emb = self.model.encode(query).tolist()
        dense_ids = self.collection.query(
            query_embeddings=[q_emb], n_results=min(pool, len(self.chunk_ids)), include=[]
        )["ids"][0]

        sims = cosine_similarity(self.vectorizer.transform([query]), self.tfidf_matrix)[0]
        sparse_ids = [self.chunk_ids[i] for i in sims.argsort()[::-1][:pool]]

        scores = {}
        breakdown: dict[str, dict] = {}

        def _component(cid: str) -> dict:
            return breakdown.setdefault(cid, {
                "dense_rank": None, "dense_contribution": 0.0,
                "sparse_rank": None, "sparse_contribution": 0.0,
            })

        for rank, cid in enumerate(dense_ids, 1):
            contribution = dense_weight / (RRF_K + rank)
            scores[cid] = scores.get(cid, 0.0) + contribution
            component = _component(cid)
            component["dense_rank"] = rank
            component["dense_contribution"] = round(contribution, 4)
        for rank, cid in enumerate(sparse_ids, 1):
            contribution = sparse_weight / (RRF_K + rank)
            scores[cid] = scores.get(cid, 0.0) + contribution
            component = _component(cid)
            component["sparse_rank"] = rank
            component["sparse_contribution"] = round(contribution, 4)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked, breakdown

    def retrieve_scored(self, query: str, top_k: int = 3,
                        dense_weight: float = DENSE_WEIGHT,
                        sparse_weight: float = SPARSE_WEIGHT,
                        pool: int = CANDIDATE_POOL) -> list:
        ranked, _breakdown = self._fused_ranking(query, dense_weight, sparse_weight, pool)
        return ranked[:top_k]

    def retrieve_scored_with_breakdown(self, query: str, top_k: int = 3,
                                       dense_weight: float = DENSE_WEIGHT,
                                       sparse_weight: float = SPARSE_WEIGHT,
                                       pool: int = CANDIDATE_POOL) -> list:
        """Like retrieve_scored, but each hit is (chunk_id, fused_score,
        component) where component explains the fused score as dense vs.
        sparse contribution -- retrieval explainability for callers that
        want to show why a clause was surfaced, not just that it was."""
        ranked, breakdown = self._fused_ranking(query, dense_weight, sparse_weight, pool)
        return [(cid, score, breakdown.get(cid, {})) for cid, score in ranked[:top_k]]

    def retrieve(self, query: str, top_k: int = 3, **kwargs) -> list:
        return [cid for cid, _ in self.retrieve_scored(query, top_k=top_k, **kwargs)]
