# RAG — Policy Agent (Retrieval-Augmented Generation)

Isolated view of the RAG workstream from the vehicle-damage insurance claim project. This
document originates on a standalone **orphan branch** with no shared history with `main`
(retrieval, generation, and evaluation components plus the data they need, with vision/YOLO and
the multi-agent orchestration graph deliberately excluded as separate workstreams).

**On `main`, this same content lives at `backend/app/rag_scripts/`** inside the full application
— all commands in §5 below are given relative to that directory (`cd backend/app/rag_scripts`
first), not repo root. `src/retrieval/*` here is genuinely used at runtime by the live app
(`backend/app/services/policy_clause_service.py` imports it directly via a `sys.path` bootstrap,
not a copy); `scripts/` is a standalone research/evaluation CLI toolkit the running app never
calls.

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
| LLM-judge eval | RAGAs (`context_precision`; `faithfulness`/`answer_relevancy`/`answer_correctness`) via `gemini-2.5-flash` | Deterministic checks catch citation-ID bookkeeping, not whether prose is actually entailed by clause text or whether a verdict is *correct* — an independent third-party judge (never one of the two models it scores) adds that layer |

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

### RAGAs evaluation

An LLM-judge layer (`scripts/ragas_eval.py`) on top of the deterministic checks above, judged
by `gemini-2.5-flash` — independent of both report-generation models, so neither ever scores
its own output:

| Metric | `llama-3.3-70b-versatile` | `openai/gpt-oss-20b` |
| :--- | ---: | ---: |
| faithfulness | 0.630 | 0.520 |
| answer_relevancy | 0.435 | 0.481 |
| answer_correctness | 0.524 | 0.593 |

| Retrieval metric | Value |
| :--- | ---: |
| context_precision (50 incidents) | **0.832** |

These sit well below the deterministic figures above (P@3 0.913, faithfulness composite 1.00) —
expected, not a regression. The deterministic retrieval eval checks whether a retrieved chunk's
regex `damage_classes` tag intersects the target set; `context_precision` asks an LLM to
actually read each chunk and judge relevance to the incident, a stricter bar. The deterministic
faithfulness checks verify citation-ID bookkeeping (does a cited `chunk_id` exist, does its type
match the verdict); `faithfulness`/`answer_correctness` check whether the model's *prose* is
actually entailed by the clause text and matches an independently-derived correct verdict
(`data/eval/reference_reports.json`, hand-written from the clause text alone — 7 of the 14
reference verdicts disagree with what the models produced, so this is not grading the models
against themselves). `gpt-oss-20b` scoring higher on correctness despite lower faithfulness than
`llama-3.3-70b-versatile` is a genuine tension between "grounded in the shown text" and "reaches
the right verdict," not noise — see `data/rag_outputs/ragas_eval.json` for the per-item breakdown.

`context_recall` is deliberately not computed: it needs a single canonical reference answer, and
this corpus has 5 differently-worded policies covering each damage class with no canonical
target document per incident (see `scripts/ragas_eval.py`'s docstring for why an early attempt
scored 0.0 on every incident).

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
data/eval/           reference_reports.json — hand-written reference verdicts for RAGAs
```

## 5. Running it

All commands below are relative to **this directory** (`backend/app/rag_scripts/` on `main`),
not the repo root — `cd` here first. `scripts/` has its own, additional dependency set beyond
`backend/requirements.txt` (heavier eval-only packages: `ragas`, `langchain-google-genai`, …).

```bash
cd backend/app/rag_scripts
pip install -r scripts/requirements.txt
cp ../../../../.env.example ../../../../.env   # from repo root; add GROQ_API_KEY, GOOGLE_API_KEY
```

**Retrieval evaluation — no API key needed, runs against the committed `data/chroma_db/` index.**
Verified working from a fresh environment; reproduces the P@3/MRR figures in §3 above (small
drift, e.g. 0.907 vs. 0.9133, is expected across dependency-version/embedding-cache differences,
not a broken pipeline):
```bash
PYTHONPATH=. python scripts/hybrid_retrieval.py --evaluate
```
**Note:** the first run against this index with the pinned `chromadb` version rewrites
`data/chroma_db/*.bin` and `chroma.sqlite3` in place (observed: `data_level0.bin` compacted from
~16.7MB to ~168KB, no change in evaluation results) — `git status` will show the fixture as
modified afterwards. This is chromadb's own on-load index migration, not data loss; `git checkout
-- data/chroma_db/` restores the originally-committed bytes if you want a clean tree, and re-running
the evaluation against the rewritten files still reproduces the same numbers.

**Rebuilding the corpus + index from the source PDFs** (`--pdf_dir`/`--output_dir` are required,
not defaulted — run `python scripts/preprocess_policy_pdfs.py --help` for the full flag list):
```bash
PYTHONPATH=. python scripts/preprocess_policy_pdfs.py \
  --pdf_dir data/policy_pdfs/synthetic \
  --output_dir data/rag_outputs
```

**Report-generation and LLM-judge evaluations** (each needs the API key noted; skips/fails
clearly without one rather than silently producing empty results):
```bash
PYTHONPATH=. python scripts/eval_report_agent.py     # faithfulness eval -- needs GROQ_API_KEY
PYTHONPATH=. python scripts/ragas_eval.py --all       # RAGAs LLM-judge -- needs GOOGLE_API_KEY
```

**Parameter sweeps and other analyses:**
```bash
PYTHONPATH=. python scripts/sweep_rag_params.py --with-chunking  # sweeps A–F
PYTHONPATH=. python scripts/sweep_significance.py                # bootstrap comparisons
PYTHONPATH=. python scripts/chunk_quality_analysis.py            # chunk-quality metrics
PYTHONPATH=. python scripts/check_real_policy_parse.py           # synthetic vs real policies
PYTHONPATH=. python scripts/ingest_user_policy.py --help         # per-user ingestion (see --help for required args)
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
5. **The RAGAs generation eval is small and single-judge.** 14 damage-class items across 10
   claims, scored by one judge model (`gemini-2.5-flash`) with no repeat runs to check
   judge-score variance. The reference verdicts it's scored against are one author's reading of
   the clause text, not an adjudicated multi-rater ground truth — treat the absolute numbers as
   directional, not precise. `context_precision`'s reference statement is still seeded from the
   same regex `damage_classes` tags as limitation #1, so it narrows but does not eliminate that
   circularity.

## 7. Notes

- `src/config.py` and `src/schemas.py` are shared with the vision track; they are included here
  because the RAG modules import them.
- `data/user_policies/` and `data/claim_pipeline/` are gitignored on `main` as transient run
  state and are therefore not on this branch.
- Full write-ups: `docs/Milestone3_Report.md` §3.5/§5.3/§9, `docs/Milestone4_Report.md` §13,
  `docs/Milestone5_Report.md` §13 — all on `main`.
