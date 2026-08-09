# RAG — Policy Agent (Retrieval-Augmented Generation)

Isolated view of the RAG workstream from the vehicle-damage insurance claim project. This is
an **orphan branch**: no shared history with `main`, containing only the retrieval, generation
and evaluation components plus the data they need.

Vision (YOLO training/inference, VehiDE/CarDD datasets) and the multi-agent orchestration graph
are deliberately excluded — they are separate workstreams on `main`.

---

## 1. What this component does

A claimant submits a damage photo and their policy PDF. The vision agent says *what* is
damaged; this component answers **what the claimant's policy actually says about it**, and
produces a grounded assessment report.

```
policy PDF
   │
   ├─▶ parse + chunk + tag        src/retrieval/policy_parser.py
   ├─▶ embed → per-user index     src/retrieval/policy_store.py
   │
detected damage classes
   │
   ├─▶ hybrid retrieval           src/retrieval/hybrid_retriever.py   (dense + sparse, RRF)
   ├─▶ coverage/exclusion split   src/retrieval/clause_retriever.py   (2 queries per class)
   │
   ├─▶ report generation          src/agents/report_agent.py          (LLM, prompted only)
   └─▶ faithfulness checks        src/evaluation/faithfulness.py      (blocking, deterministic)
```

**No trained weights.** The embedding model is used frozen and the LLMs are accessed by
prompting only. All tuning here is model selection and parameter tuning.

---

## 2. Design decisions

| Component | Choice | Why |
| :--- | :--- | :--- |
| Chunking | Structure-aware, 300 chars / 40 overlap | Splits on headings and list items first so each clause stays atomic; heading prepended as breadcrumb |
| Embedding | `all-MiniLM-L6-v2` (384-dim, frozen) | P@3 **1.00** vs **0.94** for BGE-small, at two-thirds the parameters |
| Vector store | ChromaDB, cosine, one collection per user | Identical top-1 to FAISS at this scale; metadata filtering and persistence built in |
| Sparse signal | TF-IDF, fused via weighted RRF (3:1 dense:sparse) | Catches discriminative literal terms that meaning-search misses |
| Clause retrieval | Two queries per damage class | A single query surfaces coverage and buries the exclusion that qualifies it |
| Generation | `llama-3.3-70b-versatile` (Groq) | Faithfulness composite **1.00**; `gpt-oss-20b` also scored 1.00 |
| Grounding check | Deterministic rules, not an LLM judge | Citation validity, verdict–evidence consistency, currency-figure grounding |

**Architecture change.** An earlier design inferred which of 5 catalog policies applied from
the damage profile alone. A 315-case census measured **top-1 accuracy of 0.20**, so the catalog
was removed: the claimant now uploads their own policy and each user gets a private index. The
identification problem is dissolved rather than solved.

---

## 3. Results

Retrieval, on 50 synthetic incidents against a 185-chunk / 5-policy corpus:

| Metric | Value |
| :--- | ---: |
| Mean P@3 | **0.9133** |
| Random-retrieval baseline | 0.1634 |
| Lift over random | **5.59×** |
| MRR@5 | 0.9767 |
| Zero-hit incidents | **0 / 50** |

Report faithfulness — 10 claims, both models: composite **1.00**, citation validity 1.00,
verdict–evidence consistency 1.00, **0** fabricated currency figures.

### Parameter tuning

84 configurations across 6 sweeps. The harness was validated first by reproducing all four
previously published weight-ratio data points exactly.

- **The grid is flat.** RRF_K is inert from k=1 to k=1000 (0.8933–0.9133); candidate pool is
  flat above 5; the 5×5 interaction grid spans 0.8933–0.9200 with no ridge.
- **Only one of 15 configurations is statistically distinguishable from production** —
  sparse-only, and it is *worse* (0.7400, 95% CI [−0.2400, −0.1067]). Every other confidence
  interval spans zero.
- **Two parameters are inactive.** `MIN_CLAUSE_SCORE = 0.01` sits below the analytic minimum
  fused score of 0.0125 and filters 0 of 55 returned clauses. `DEDUP_THRESHOLD = 0.90` removes
  0 chunks at every threshold from 0.80 to 1.00 and at all 8 chunk sizes.
- **Larger chunks score higher for the wrong reason.** Raw P@3 rises with chunk size but so
  does the random baseline, so lift *falls* (7.24× at 150 chars → 4.61× at 1000), while chunks
  mixing a coverage grant with its exclusion nearly double (9.9% → 26.9%).

At n=50, mean P@3 moves only in quanta of 1/150 = 0.00667, so **the evaluation set, not the
configuration, is the binding constraint**.

---

## 4. Layout

```
src/retrieval/       policy_parser · policy_store · hybrid_retriever · clause_retriever
src/evaluation/      faithfulness.py — deterministic grounding checks
src/agents/          report_agent.py — LLM report generation
src/config.py        all RAG parameters (lines 73–91); shared with the vision track on main
scripts/             ingestion, retrieval, report generation, evaluation, parameter sweeps
data/policy_pdfs/    5 synthetic policies (tuned on) + 3 real IRDAI-filed policies
data/rag_outputs/    chunk corpus, evaluation results, sweep results
data/chroma_db/      prebuilt 185-chunk index — the retrieval-quality regression bed
```

## 5. Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY for report generation

PYTHONPATH=. python3 scripts/preprocess_policy_pdfs.py     # rebuild the corpus + index
PYTHONPATH=. python3 scripts/hybrid_retrieval.py --evaluate # 50-incident retrieval eval
PYTHONPATH=. python3 scripts/ingest_user_policy.py          # per-user ingestion
PYTHONPATH=. python3 scripts/eval_report_agent.py           # faithfulness eval (needs API key)

PYTHONPATH=. python3 scripts/sweep_rag_params.py --with-chunking  # sweeps A–F
PYTHONPATH=. python3 scripts/sweep_significance.py                # bootstrap comparisons
PYTHONPATH=. python3 scripts/chunk_quality_analysis.py            # chunk-quality metrics
PYTHONPATH=. python3 scripts/check_real_policy_parse.py           # synthetic vs real policies
```

Everything is deterministic; the bootstrap is seeded (`seed=20260807`). Sweeps A–E finish in
seconds by caching dense and sparse rankings once per query at depth 100 — no fusion parameter
can change either ranking, only how the two are combined.

---

## 6. Limitations

State these before someone else finds them.

1. **Relevance labels are circular.** `data/clause_groundtruth.json` is not hand-labelled; its
   `damage_classes` are the same regex auto-tags the chunker writes onto each chunk (verified
   identical for all 185). Scores measure retrieval against the tagger, not a human adjudicator.
2. **The tuning set is the test set.** The incident that motivated hybrid retrieval is one of
   the 50 used to score it, and the 3:1 ratio was chosen on the same 50. No held-out split.
3. **Measured on synthetic policies that suit the method.** The 5 synthetic policies each
   mention all 6 damage types; the 3 real IRDAI policies mention only 2–4, and 96–98% of their
   chunks carry no damage-class tag (vs 15–67% synthetic). Not a tagger defect — real motor
   policies use generic "loss or damage" wording — but retrieval signal on a real uploaded
   policy is thinner than these figures imply.
4. **The per-user path has no metrics of its own.** Every figure above comes from the shared
   5-policy corpus. Under per-user indexing TF-IDF is fit on a single document, a materially
   different sparse signal. Measuring it is the highest-value outstanding work.

## 7. Notes

- `src/config.py` and `src/schemas.py` are shared with the vision track; they are included here
  because the RAG modules import them.
- `data/user_policies/` and `data/claim_pipeline/` are gitignored on `main` as transient run
  state and are therefore not on this branch.
- Full write-ups: `docs/Milestone3_Report.md` §3.5/§5.3/§9, `docs/Milestone4_Report.md` §13,
  `docs/Milestone5_Report.md` §13 — all on `main`.
