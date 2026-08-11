"""
Regenerate the 10 mile3 claim reports through the CURRENT production src/
pipeline instead of the frozen scripts/report_context.py + scripts/report_agent.py
snapshot the original mile3 artifacts came from.

Why this exists: src/retrieval/clause_retriever.py gained a general-coverage
fallback query (a class-specific coverage query can be outranked by an
unrelated chunk that shares more surface vocabulary with the query than the
real operative clause does -- confirmed for crack/broken_lamp against
policy_3_quickclaim_general). That fix lives in src/, not in the frozen
scripts/ snapshot, so seeing its effect on generation quality means running
the actual production retrieval + generation path, not re-running the old
script.

Only the retrieval-and-generation half changes here. claim_id,
incident_narrative, detections, and escalation are reused verbatim from
data/rag_outputs/mile3/payloads_all.json -- those are inputs to retrieval,
not outputs of it, so there is nothing to regenerate about them.

Generator model: gemini-2.5-flash, not src/config.py's REPORT_MODEL
(llama-3.3-70b-versatile). Groq's free-tier daily token quota for that model
was already exhausted mid-run by this same script (4/10 claims got through
before every remaining request hit "Rate limit reached... tokens per day"),
and it does not reset fast enough for this to finish in one sitting.
generate_report() from src/agents/report_agent.py is reused completely
unchanged -- it just takes an OpenAI-SDK client and a model name, and
Gemini exposes an OpenAI-compatible endpoint, so only the client
construction differs, not the report-generation logic itself.

Caveat worth being explicit about: scripts/ragas_eval.py's judge is also
gemini-2.5-flash. With Gemini as both generator and judge, faithfulness and
answer_relevancy on this run are no longer scored by a fully independent
third party for this half of the comparison -- a model judging its own
output can be more lenient about its own claims than a neutral judge would
be. answer_correctness is less exposed to this, since it's scored against
data/eval/reference_reports.json (an independently hand-written reference),
not against the judge's own opinion of the response.

Run:
    PYTHONPATH=. python3 scripts/regen_via_src_pipeline.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.policy_store import ingest_policy, has_policy, chunks_tsv_path
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.clause_retriever import ClauseRetriever, build_context_bundle
from src.agents.report_agent import generate_report
from src.evaluation.faithfulness import check_report

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OLD_PAYLOADS_PATH = ROOT / "data" / "rag_outputs" / "mile3" / "payloads_all.json"
PDF_DIR = ROOT / "data" / "policy_pdfs" / "synthetic"
OUT_DIR = ROOT / "data" / "rag_outputs" / "mile3_src_pipeline"

GENERATION_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
REPORT_FILE = f"reports_{GENERATION_MODEL.replace('/', '_')}.json"


def get_gemini_client() -> OpenAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. Add it to .env (see .env.example).")
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

USER_IDS = [
    "policy_1_bharat_suraksha",
    "policy_2_safedrive_assurance",
    "policy_3_quickclaim_general",
    "policy_4_autoguard_premium",
    "policy_5_valuemotor",
]


def ensure_ingested():
    for uid in USER_IDS:
        if has_policy(uid):
            print(f"  {uid}: already ingested")
            continue
        pdf = PDF_DIR / f"{uid}.pdf"
        print(f"  {uid}: ingesting {pdf.name} ...")
        manifest = ingest_policy(pdf, uid)
        print(f"    -> {manifest.n_chunks} chunks")


def load_chunk_damage_classes(user_id: str) -> dict:
    out = {}
    import csv
    with open(chunks_tsv_path(user_id), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["chunk_id"]] = [c for c in row["damage_classes"].split(",") if c]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Ensuring the 5 synthetic policies are ingested at the src/ production location...")
    ensure_ingested()

    with open(OLD_PAYLOADS_PATH) as f:
        old_payloads = json.load(f)

    # Resumable: Groq's daily token quota can run out mid-run (it did, on
    # the first attempt at this exact script). Reuse any already-succeeded
    # payload+report pairs from a prior run instead of re-spending quota to
    # regenerate claims that already came back "ok".
    payloads_path = OUT_DIR / "payloads_all.json"
    reports_path = OUT_DIR / REPORT_FILE
    done_payloads, done_reports = {}, {}
    if payloads_path.exists() and reports_path.exists():
        with open(payloads_path) as f:
            prev_payloads = {p["claim_id"]: p for p in json.load(f)}
        with open(reports_path) as f:
            prev_reports = {r["claim_id"]: r for r in json.load(f)}
        for cid, r in prev_reports.items():
            if r.get("ok"):
                done_payloads[cid] = prev_payloads[cid]
                done_reports[cid] = r
        if done_reports:
            print(f"Resuming: {len(done_reports)}/{len(old_payloads)} claims already "
                  f"succeeded in a prior run, skipping those.")

    client = get_gemini_client()
    retrievers = {}
    new_payloads, reports = [], []

    print(f"\nRegenerating {len(old_payloads)} claims via src/ retrieval + src/ report_agent "
          f"({GENERATION_MODEL})...")
    for old in old_payloads:
        claim_id = old["claim_id"]
        if claim_id in done_reports:
            new_payloads.append(done_payloads[claim_id])
            reports.append(done_reports[claim_id])
            print(f"  {claim_id}: ok (reused from prior run)")
            continue

        user_id = old["policy"]["user_id"]
        if user_id not in retrievers:
            retrievers[user_id] = HybridRetriever(user_id)
        clause_retriever = ClauseRetriever(retrievers[user_id])

        damage_classes = sorted({d["class_name"] for d in old["detections"]})
        clauses_by_class = {cls: clause_retriever.get_clauses(cls) for cls in damage_classes}

        bundle = build_context_bundle(
            claim_id=claim_id,
            user_id=user_id,
            incident_narrative=old["incident_narrative"],
            detections=old["detections"],
            clauses_by_class=clauses_by_class,
            escalation=old["escalation"],
        )
        new_payloads.append(bundle)

        result = generate_report(bundle, model=GENERATION_MODEL, client=client)
        result["claim_id"] = claim_id
        result["model"] = GENERATION_MODEL
        reports.append(result)
        print(f"  {claim_id}: {'ok' if result['ok'] else 'FAILED: ' + str(result['error'])}")

        # Save after every claim, not just at the end, so a mid-run quota
        # exhaustion (as above) doesn't lose already-succeeded work.
        with open(payloads_path, "w") as f:
            json.dump(new_payloads, f, indent=2)
        with open(reports_path, "w") as f:
            json.dump(reports, f, indent=2)

    print("\nDeterministic faithfulness check (src/evaluation/faithfulness.py):")
    chunk_classes_cache = {}
    n_passed = 0
    for payload, r in zip(new_payloads, reports):
        if not r["ok"]:
            print(f"  {r['claim_id']}: SKIPPED (generation failed)")
            continue
        uid = payload["policy"]["user_id"]
        if uid not in chunk_classes_cache:
            chunk_classes_cache[uid] = load_chunk_damage_classes(uid)
        check = check_report(payload, r["parsed"], chunk_classes_cache[uid])
        n_passed += check["passed"]
        status = "PASS" if check["passed"] else f"HARD FAILURES: {check['hard_failures']}"
        print(f"  {r['claim_id']}: {status}")
    print(f"\n{n_passed}/{len(reports)} claims passed deterministic faithfulness checks")

    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
