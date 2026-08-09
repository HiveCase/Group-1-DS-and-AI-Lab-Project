"""
Generates the preliminary claim report from a context bundle, via a single
LLM call to Groq Cloud's OpenAI-compatible API.

One model, not two. The scripts/ version ran llama-3.3-70b-versatile and
gpt-oss-20b head-to-head on every payload because it was a comparison
harness deciding which to adopt; both scored a perfect 1.0 composite on the
mechanical faithfulness eval, with llama-3.3-70b marginally ahead on the
tie-breaking soft flags. That comparison is finished, so this pipeline calls
the winner once. The model id is a config value (src/config.py REPORT_MODEL)
precisely because that decision may be revisited.

The grounding rules in the system prompt exist because Milestone 1 flagged
this exact risk: "General-purpose LLMs can produce plausible
insurance-related text, but without access to the specific policy document,
they hallucinate coverage entitlements, cite incorrect exclusions, or
fabricate deductible values." So the prompt forbids inventing clause
content, forbids stating any rupee figure (no financial reasoning exists in
this pipeline -- a number here would be fabricated by construction), and
requires every verdict to cite the chunk_ids it rests on. Those citations
are what makes src/evaluation/faithfulness.py able to mechanically verify
grounding instead of taking the model's word for it.
"""
import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    GROQ_BASE_URL,
    REPORT_MAX_RETRIES,
    REPORT_MODEL,
    REPORT_TEMPERATURE,
)

log = logging.getLogger(__name__)
load_dotenv()

SYSTEM_PROMPT = """You are a preliminary motor-insurance claims assessment assistant.

You receive a JSON claim payload with: an incident narrative, model-detected \
vehicle damage (class, severity, confidence), the claimant's own uploaded \
policy document, and, per damage class, retrieved policy clauses split into \
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
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to .env (see .env.example) -- the report "
            "step calls Groq Cloud and cannot run offline."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def generate_report(context_bundle: dict, model: str = REPORT_MODEL,
                    client: OpenAI = None, max_retries: int = REPORT_MAX_RETRIES) -> dict:
    """Call the LLM and return {"ok", "parsed", "raw", "error", "model"}.

    Returns a failure dict rather than raising: a failed report is a claim
    outcome the graph must route on (to human review), not a crash that
    loses the checkpointed run.
    """
    client = client or get_client()
    user_content = json.dumps(context_bundle, indent=2)
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=REPORT_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return {"ok": True, "parsed": json.loads(raw), "raw": raw,
                    "error": None, "model": model}
        except json.JSONDecodeError as e:
            # response_format=json_object makes this rare, but a truncated
            # response still lands here -- worth one more attempt.
            last_err = f"JSON parse error: {e}"
            log.warning("Attempt %d/%d: %s", attempt + 1, max_retries, last_err)
        except Exception as e:
            last_err = str(e)
            log.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, last_err)
            time.sleep(2 * (attempt + 1))  # linear backoff on rate limits

    return {"ok": False, "parsed": None, "raw": None, "error": last_err, "model": model}
