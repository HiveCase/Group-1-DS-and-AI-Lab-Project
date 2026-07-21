"""
Report Agent: turns a context bundle (scripts/report_context.py) into a
structured preliminary claim report via an LLM, called through Groq Cloud's
OpenAI-compatible API.

Two models are run head-to-head per payload for comparison (see
docs/rag_support_mile3.md): llama-3.3-70b-versatile and openai/gpt-oss-20b.

Grounding rules (system prompt below) exist because Milestone 1 flagged this
exact risk: "General-purpose LLMs can produce plausible insurance-related
text, but without access to the specific policy document, they hallucinate
coverage entitlements, cite incorrect exclusions, or fabricate deductible
values." The prompt forbids inventing any clause content, forbids stating a
rupee claim amount (out of scope -- see the qualitative-verdict-only
decision in docs/rag_support_mile3.md), and requires every verdict to cite
the chunk_id(s) it rests on, so scripts/eval_report_agent.py can mechanically
check the citation actually exists in the payload rather than trusting the
model's word for it.
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PAYLOADS_PATH = ROOT / "data" / "rag_outputs" / "mile3" / "payloads_all.json"
OUT_DIR = ROOT / "data" / "rag_outputs" / "mile3" / "reports"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-20b"]

SYSTEM_PROMPT = """You are a preliminary motor-insurance claims assessment assistant.

You receive a JSON claim payload with: an incident narrative, YOLO-detected \
vehicle damage (class, severity, confidence), an auto-selected policy \
document, and, per damage class, retrieved policy clauses split into \
"coverage" and "exclusion_or_condition".

Rules you must follow exactly:
1. Only use clause text present in payload.policy.clauses. Never invent, \
assume, or recall from general knowledge any coverage term, exclusion, \
deductible, depreciation percentage, or IDV figure not present in the \
given clause text.
2. Do not state or compute any rupee/currency claim amount. This is a \
qualitative coverage assessment only -- no financial figures exist in this \
pipeline yet.
3. For each damage_class in the payload, output exactly one verdict: \
"covered", "excluded", "conditional", or "needs_review".
   - Use "needs_review" if coverage_clause_found is false for that class, \
or if the retrieved clauses do not let you decide confidently.
   - Use "conditional" if coverage depends on a condition (e.g. "only if \
concurrent vehicle damage present") that may or may not be satisfied by \
the incident narrative as given -- state the condition explicitly.
   - Every verdict must cite the chunk_id(s) of the clause(s) it rests on \
in cited_chunk_ids. Do not cite a chunk_id that was not given to you.
4. If payload.escalation.needs_human_review is true, set \
escalate_to_human=true and explain why in escalation_reason.
5. Output ONLY valid JSON, matching this schema, no prose outside the JSON:
{
  "claim_id": string,
  "policy_doc_id": string,
  "items": [
    {"damage_class": string, "verdict": string, "rationale": string, "cited_chunk_ids": [string]}
  ],
  "overall_recommendation": string,
  "escalate_to_human": boolean,
  "escalation_reason": string or null
}
"""


def get_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set (expected in .env)")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def generate_report(client: OpenAI, model: str, payload: dict, max_retries: int = 3) -> dict:
    user_content = json.dumps(payload, indent=2)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return {"ok": True, "raw": raw, "parsed": json.loads(raw)}
        except json.JSONDecodeError as e:
            last_err = f"JSON parse error: {e}"
        except Exception as e:
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "error": last_err, "raw": None, "parsed": None}


def main():
    with open(PAYLOADS_PATH) as f:
        payloads = json.load(f)

    client = get_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model in MODELS:
        model_results = []
        for payload in payloads:
            print(f"[{model}] generating report for {payload['claim_id']}...")
            result = generate_report(client, model, payload)
            result["claim_id"] = payload["claim_id"]
            result["model"] = model
            model_results.append(result)
        safe_model_name = model.replace("/", "_")
        with open(OUT_DIR / f"reports_{safe_model_name}.json", "w") as f:
            json.dump(model_results, f, indent=2)
        all_results[model] = model_results
        n_ok = sum(1 for r in model_results if r["ok"])
        print(f"[{model}] {n_ok}/{len(model_results)} reports generated successfully\n")

    with open(OUT_DIR / "reports_all.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved all reports -> {OUT_DIR}")


if __name__ == "__main__":
    main()
