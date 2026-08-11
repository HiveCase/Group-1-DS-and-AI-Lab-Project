# RAG — Policy Agent (Retrieval-Augmented Generation)

The retrieval, generation, and evaluation component of the vehicle-damage insurance claim
pipeline. A claimant submits a damage photo and their policy PDF. The vision agent (`src/agents/`
detection nodes) says *what* is damaged; this component answers **what the claimant's policy
actually says about it**, and produces a grounded assessment report.

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

Wired together as LangGraph nodes in `src/agents/graph.py`; this doc covers the retrieval/
generation/evaluation modules those nodes call, developed and tuned in isolation before
integration.

**No trained weights.** The embedding model is used frozen and the LLMs are accessed by
prompting only. All tuning here is model selection and parameter tuning.

---

## 1. Design decisions

| Component | Choice | Why |
| :--- | :--- | :--- |
| Chunking | Structure-aware, 300 chars / 40 overlap | Splits on headings and list items first so each clause stays atomic; heading prepended as breadcrumb |
| Embedding | `all-MiniLM-L6-v2` (384-dim, frozen) | P@3 **1.00** vs **0.94** for BGE-small, at two-thirds the parameters |
| Vector store | ChromaDB, cosine, one collection per user | Identical top-1 to FAISS at this scale; metadata filtering and persistence built in |
| Sparse signal | TF-IDF, fused via weighted RRF (3:1 dense:sparse) | Catches discriminative literal terms that meaning-search misses |
| Clause retrieval | Two queries per damage class, plus a class-agnostic general-coverage fallback merged into the coverage pool | A single class-specific query can be outranked by an unrelated coverage-typed chunk sharing more surface vocabulary with the query than the real operative clause — see §3.2 |
| Generation | `llama-3.3-70b-versatile` (Groq) | Faithfulness composite **1.00**; `gpt-oss-20b` also scored 1.00 |
| Grounding check | Deterministic rules, not an LLM judge | Citation validity, verdict–evidence consistency, currency-figure grounding |
| LLM-judge eval | RAGAs (`context_precision`; `faithfulness`/`answer_relevancy`/`answer_correctness`) via `gemini-2.5-flash` | Deterministic checks catch citation-ID bookkeeping, not whether prose is actually entailed by clause text or whether a verdict is *correct* — an independent third-party judge (never one of the models it scores, except where noted in §3.3) adds that layer |

**Architecture change.** An earlier design inferred which of 5 catalog policies applied from
the damage profile alone. A 315-case census measured **top-1 accuracy of 0.20**, so the catalog
was removed: the claimant now uploads their own policy and each user gets a private index. The
identification problem is dissolved rather than solved.

---

## 2. Results — shared 5-policy corpus

Retrieval, on 50 synthetic incidents against a 185-chunk / 5-policy corpus:

| Metric | Value |
| :--- | ---: |
| Mean P@3 | **0.9133** |
| Random-retrieval baseline | 0.1634 |
| Lift over random | **5.59×** |
| MRR@5 | 0.9767 |
| Zero-hit incidents | **0 / 50** |

Report faithfulness — 10 claims, both models, generated through the pre-fix retrieval path
described in §3.2: composite **1.00**, citation validity 1.00, verdict–evidence consistency
1.00, **0** fabricated currency figures.

### 2.1 RAGAs evaluation (pre-fix)

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
against themselves).

`context_recall` is deliberately not computed: it needs a single canonical reference answer, and
this corpus has 5 differently-worded policies covering each damage class with no canonical
target document per incident (see `scripts/ragas_eval.py`'s docstring for why an early attempt
scored 0.0 on every incident).

### 2.2 Parameter tuning

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

## 3. Post-fix: the per-user production path, measured

Limitation #4 below (pre-fix version) flagged that every figure above comes from the shared
corpus and the per-user path had no metrics of its own. This section closes part of that gap.

### 3.1 What was found

Running the 10 milestone-3 claims through the RAGAs generation eval (§2.1) surfaced a real bug,
not just a low score. `CLAIM_04_multi_pileup_severe`'s `crack` and `broken_lamp` items had
`coverage_clause_found=True`, but the only chunk in that bucket was an unrelated tyre-damage
condition clause — the policy's actual general operative clause ("accidental loss or damage to
the vehicle insured...") was never retrieved for either class. The model's rationale asserted
*"the policy covers cracks"* with nothing behind that claim. The deterministic checks (citation-
ID bookkeeping) couldn't catch this: the citation existed and its type matched, it was just the
wrong clause.

### 3.2 The fix

`src/retrieval/clause_retriever.py`'s `get_clauses()` now also runs a class-agnostic query for
the policy's general coverage grant and merges its results into the class-specific coverage
candidates (max-score dedup, additive only — never displaces an existing hit). Verified against
the live corpus: `policy_3_quickclaim_general`'s operative clause, previously invisible to
`crack`/`broken_lamp`, now ranks first for both. Swept all 5 policies × 6 damage classes — every
bucket now has a non-empty, sane coverage set (1–5 hits).

### 3.3 Re-measuring through the production path

`scripts/regen_via_src_pipeline.py` ingests the 5 policies at the actual `src/` production
location (`data/claim_pipeline/policies/`, not the frozen `scripts/report_context.py` snapshot
the original milestone-3 artifacts came from) and regenerates all 10 claims through the current
`src/retrieval/clause_retriever.py` + `src/agents/report_agent.py`. All 10 now pass the
deterministic faithfulness checks (10/10, up from partial passes pre-fix), and `CLAIM_04`'s
crack/broken_lamp items now cite the real operative clause and reason from it correctly instead
of asserting ungrounded coverage.

Generation for this run used `gemini-2.5-flash` rather than `src/config.py`'s configured
`REPORT_MODEL` (`llama-3.3-70b-versatile`): Groq's free-tier daily token quota for that model was
exhausted mid-session by this same evaluation work and does not reset fast enough to finish a
10-claim regeneration in one sitting. `generate_report()` itself is unchanged — Gemini exposes an
OpenAI-compatible endpoint, so only the client construction differs.

| Metric | Pre-fix, `llama-3.3-70b-versatile` (Groq gen, Gemini judge) | Post-fix, `gemini-2.5-flash` (Gemini gen, Gemini judge) |
| :--- | ---: | ---: |
| faithfulness | 0.630 | 0.475 |
| answer_relevancy | 0.435 | 0.489 |
| answer_correctness | 0.524 | **0.659** |

**Read this carefully, not as a clean before/after.** Two things changed at once: the retrieval
fix, and the generator model (forced by the Groq quota exhaustion). `answer_correctness`
improving is consistent with the concrete fix in §3.1 — the model now has real grounding for
items it previously had to bluff — but `faithfulness` dropping cannot be cleanly attributed to
the retrieval fix given the simultaneous model swap. There is also a genuine independence gap in
this specific comparison: `gemini-2.5-flash` is both the generator and the RAGAs judge for the
post-fix numbers, so faithfulness/answer_relevancy here are not judged by a fully independent
third party the way the pre-fix, cross-provider comparison was. Re-running with
`llama-3.3-70b-versatile` once Groq quota allows, to isolate the retrieval fix from the model
choice, is the natural next step.

Full artifacts: `data/rag_outputs/mile3_src_pipeline/` (payloads, reports, RAGAs results).

---

## 4. Layout

```
src/retrieval/       policy_parser · policy_store · hybrid_retriever · clause_retriever
src/evaluation/      faithfulness.py — deterministic grounding checks
src/agents/          report_agent.py — LLM report generation; graph.py/nodes.py — orchestration
src/config.py        all RAG parameters; shared with the vision track
scripts/             ingestion, retrieval, report generation, evaluation, parameter sweeps
scripts/regen_via_src_pipeline.py  regenerate claims through the current src/ production path
data/policy_pdfs/    5 synthetic policies (tuned on) + 3 real IRDAI-filed policies
data/rag_outputs/    chunk corpus, evaluation results, sweep results
data/chroma_db/      prebuilt 185-chunk index — the retrieval-quality regression bed
data/eval/           reference_reports.json — hand-written reference verdicts for RAGAs
data/claim_pipeline/ per-user policy indices + claim records (gitignored; regenerable local state)
```

## 5. Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY for report generation, GOOGLE_API_KEY for RAGAs

PYTHONPATH=. python3 scripts/preprocess_policy_pdfs.py     # rebuild the corpus + index
PYTHONPATH=. python3 scripts/hybrid_retrieval.py --evaluate # 50-incident retrieval eval
PYTHONPATH=. python3 scripts/ingest_user_policy.py          # per-user ingestion
PYTHONPATH=. python3 scripts/eval_report_agent.py           # faithfulness eval (needs API key)
PYTHONPATH=. python3 scripts/ragas_eval.py --all             # LLM-judge eval (needs GOOGLE_API_KEY)
PYTHONPATH=. python3 scripts/regen_via_src_pipeline.py       # re-run claims through src/ production path

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
4. **The per-user path is now partially measured, not fully.** §3 regenerated the 10 milestone-3
   claims through the actual per-user production path, but that is still only 10 claims / 14
   items on the same 5 synthetic policies used for tuning — not an independent per-user eval set,
   and confounded by the Groq→Gemini generator swap (§3.3). A held-out per-user incident set,
   scored with a consistent generator model, is the highest-value outstanding work.
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
- `data/user_policies/` and `data/claim_pipeline/` are gitignored as transient, regenerable
  per-user runtime state (policy uploads + the vector index built from them, and claim records) —
  not shipped in the repository beyond a small `data/claim_pipeline/claims/` and `policies/`
  demo fixture (`alice`/`bob`) used to exercise the graph's checkpointing behavior.
- Full write-ups: `docs/Milestone3_Report.md` §3.5/§5.3/§9, `docs/Milestone4_Report.md` §13,
  `docs/Milestone5_Report.md` §13.4.
