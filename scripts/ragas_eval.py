"""
RAGAs LLM-judge evaluation, layered on top of the deterministic checks in
scripts/hybrid_retrieval.py (retrieval P@3/MRR against regex-tag ground
truth) and scripts/eval_report_agent.py (citation-ID grounding). Those are
mechanical/tag-based; this pass asks an LLM to actually read the retrieved
text and judge relevance/entailment/correctness, which catches things
tag-matching and citation-ID bookkeeping structurally cannot: a chunk that's
genuinely on-topic despite a missing/wrong damage_classes tag, or a
rationale that cites a valid chunk_id but mischaracterizes what the clause
actually says.

Two datasets, two metric sets:

  Retrieval (50 incidents, shared 5-policy corpus,
  data/rag_outputs/eval/incident_descriptions.json):
    context_precision only -- an LLM judges each retrieved chunk
    individually against a reference statement built from the incident's
    ground-truth damage_classes tags. That reference is still tag-seeded
    (which damage class *should* matter is still taken from the same regex
    tags README limitation #1 flags), so this does not fully escape the
    circularity -- but the relevance judgment itself is now a semantic read
    of the chunk text by an LLM, not a regex-tag set intersection, which is
    the part that can silently misjudge a correctly-tagged-but-irrelevant
    chunk or an untagged-but-relevant one.

    context_recall is deliberately NOT included here: it works by breaking
    a reference *answer* into sentences and checking each is attributable
    to the retrieved context, which presupposes a single canonical answer.
    This corpus has 5 differently-worded policies covering the same
    damage class with no canonical target document per incident, so
    "recall against one reference" isn't a coherent target -- an earlier
    attempt using a meta-description as reference scored 0.0 on every
    incident (the description sentence is never itself present in any
    clause, so it can never be judged attributable). The existing P@3/MRR
    eval in scripts/hybrid_retrieval.py already covers a recall-like
    signal against the full multi-document tag set.

  Generation (10 mile3 claims x 2 models, 14 damage-class items each):
    faithfulness, answer_relevancy, answer_correctness -- judged against
    data/eval/reference_reports.json, a hand-written reference verdict +
    rationale per (claim_id, damage_class), derived independently from the
    clause text actually offered to the model in that payload -- not
    copied from either model's output. 7 of the 14 reference verdicts
    disagree with what the models produced (see that file's
    reference_rationale fields), so this is not a rubber stamp.

Judge: gemini-2.5-flash for both phases and both report-generation models.
An early version cross-judged between the two Groq models that also
generate the reports (llama-3.3-70b-versatile / openai/gpt-oss-20b) to
avoid a model rating its own output, but their free-tier TPD/TPM caps
(100K-500K tokens/day, 6K-12K tokens/min) were too tight for a 50-incident
+ 28-item run: a first pass exhausted llama-3.3-70b-versatile's 100K TPD
outright, and a throttled retry against llama-3.1-8b-instant (500K TPD but
only 6K TPM) still silently NaN'd 18-34/50 retrieval rows on persistent
per-request timeouts. Gemini is a fully independent third-party judge (it
generated neither model's reports) with enough headroom to run the full
scope in one pass, which also sidesteps the cross-judge quota juggling
entirely.

Needs GROQ_API_KEY (unused by this script directly, but scripts/report_agent.py
needs it to have produced data/rag_outputs/mile3/reports/*.json in the first
place) and GOOGLE_API_KEY (the RAGAs judge) in .env.

Run:
    PYTHONPATH=. python3 scripts/ragas_eval.py --retrieval
    PYTHONPATH=. python3 scripts/ragas_eval.py --generation
    PYTHONPATH=. python3 scripts/ragas_eval.py --all
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
)

from hybrid_retrieval import HybridRetriever

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_PATH = ROOT / "data" / "rag_outputs" / "eval" / "incident_descriptions.json"
PAYLOADS_PATH = ROOT / "data" / "rag_outputs" / "mile3" / "payloads_all.json"
REPORTS_DIR = ROOT / "data" / "rag_outputs" / "mile3" / "reports"
REFERENCE_REPORTS_PATH = ROOT / "data" / "eval" / "reference_reports.json"
OUT_PATH = ROOT / "data" / "rag_outputs" / "ragas_eval.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 5
JUDGE_MODEL = "gemini-2.5-flash"

MODEL_FILES = {
    "llama-3.3-70b-versatile": "reports_llama-3.3-70b-versatile.json",
    "openai/gpt-oss-20b": "reports_openai_gpt-oss-20b.json",
}

RUN_CONFIG = RunConfig(max_workers=4, timeout=120)


def _judge_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0))


def _embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))


# --------------------------------------------------------------------------
# Retrieval-side dataset
# --------------------------------------------------------------------------

def build_retrieval_dataset() -> EvaluationDataset:
    with open(INCIDENTS_PATH) as f:
        incidents = json.load(f)

    retriever = HybridRetriever()
    samples = []
    for inc in incidents:
        scored = retriever.retrieve_scored(inc["description"], top_k=RETRIEVAL_TOP_K)
        contexts = [retriever.chunk_meta[cid]["text"] for cid, _ in scored]
        reference = (
            f"Relevant policy clauses should describe coverage, exclusions, or "
            f"conditions applicable to {', '.join(inc['damage_classes'])} damage "
            f"to the vehicle, in relation to this incident: {inc['description']}"
        )
        samples.append({
            "user_input": inc["description"],
            "retrieved_contexts": contexts,
            "reference": reference,
        })
    return EvaluationDataset.from_list(samples)


def run_retrieval_eval() -> dict:
    dataset = build_retrieval_dataset()
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference()],
        llm=_judge_llm(),
        embeddings=_embeddings(),
        run_config=RUN_CONFIG,
    )
    df = result.to_pandas()
    return {
        "n_incidents": len(df),
        "mean_context_precision": round(float(df["llm_context_precision_with_reference"].mean()), 3),
        "judge_model": JUDGE_MODEL,
        "per_incident": df[["user_input", "llm_context_precision_with_reference"]]
            .to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# Generation-side dataset
# --------------------------------------------------------------------------

def build_generation_dataset(report_path, payloads_by_id: dict, reference_by_key: dict) -> tuple:
    with open(report_path) as f:
        reports = json.load(f)

    samples, keys = [], []
    for r in reports:
        if not r["ok"]:
            continue
        payload = payloads_by_id[r["claim_id"]]
        for item in r["parsed"]["items"]:
            key = (r["claim_id"], item["damage_class"])
            ref = reference_by_key.get(key)
            if ref is None:
                continue  # no hand-written reference for this item

            bucket = payload["policy"]["clauses"].get(
                item["damage_class"], {"coverage": [], "exclusion_or_condition": []}
            )
            contexts = [c["text"] for c in bucket["coverage"] + bucket["exclusion_or_condition"]]

            samples.append({
                "user_input": f"{payload['incident_narrative']} (damage class: {item['damage_class']})",
                "retrieved_contexts": contexts,
                "response": f"Verdict: {item.get('verdict')}. {item.get('rationale', '')}",
                "reference": f"Verdict: {ref['reference_verdict']}. {ref['reference_rationale']}",
            })
            keys.append(key)
    return EvaluationDataset.from_list(samples), keys


def run_generation_eval(payloads_path=PAYLOADS_PATH, reports_dir=REPORTS_DIR,
                         model_files=None) -> dict:
    model_files = MODEL_FILES if model_files is None else model_files
    with open(payloads_path) as f:
        payloads_by_id = {p["claim_id"]: p for p in json.load(f)}
    with open(REFERENCE_REPORTS_PATH) as f:
        reference_by_key = {
            (r["claim_id"], r["damage_class"]): r for r in json.load(f)
        }

    embeddings = _embeddings()
    results = {}
    for model_name, report_file in model_files.items():
        dataset, keys = build_generation_dataset(
            reports_dir / report_file, payloads_by_id, reference_by_key)
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), AnswerCorrectness()],
            llm=_judge_llm(),
            embeddings=embeddings,
            run_config=RUN_CONFIG,
        )
        df = result.to_pandas()
        df.insert(0, "damage_class", [k[1] for k in keys])
        df.insert(0, "claim_id", [k[0] for k in keys])

        results[model_name] = {
            "n_items": len(df),
            "judge_model": JUDGE_MODEL,
            "mean_faithfulness": round(float(df["faithfulness"].mean()), 3),
            "mean_answer_relevancy": round(float(df["answer_relevancy"].mean()), 3),
            "mean_answer_correctness": round(float(df["answer_correctness"].mean()), 3),
            "per_item": df[[
                "claim_id", "damage_class", "faithfulness", "answer_relevancy", "answer_correctness",
            ]].to_dict(orient="records"),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="RAGAs LLM-judge evaluation")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval-side eval only")
    parser.add_argument("--generation", action="store_true", help="Run generation-side eval only")
    parser.add_argument("--all", action="store_true", help="Run both (default if no flag given)")
    args = parser.parse_args()
    run_retrieval = args.retrieval or args.all or not (args.retrieval or args.generation)
    run_generation = args.generation or args.all or not (args.retrieval or args.generation)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {}

    if run_retrieval:
        print("Running retrieval-side RAGAs eval (context_precision)...")
        out["retrieval"] = run_retrieval_eval()
        print(f"  mean_context_precision={out['retrieval']['mean_context_precision']}")
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)

    if run_generation:
        print("Running generation-side RAGAs eval (faithfulness, answer_relevancy, answer_correctness)...")
        out["generation"] = run_generation_eval()
        for model_name, r in out["generation"].items():
            print(f"  {model_name} (judged by {r['judge_model']}): "
                  f"faithfulness={r['mean_faithfulness']} "
                  f"answer_relevancy={r['mean_answer_relevancy']} "
                  f"answer_correctness={r['mean_answer_correctness']}")
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)

    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
