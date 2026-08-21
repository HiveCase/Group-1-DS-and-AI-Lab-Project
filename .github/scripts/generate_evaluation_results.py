"""Runs both on-demand evaluations and writes a single evaluation-results.json.

Two independent, already-committed-data evaluations (no secrets, no
dataset download):
  1. YOLO damage-detection smoke test (yolo_smoke_test.py, same directory)
     -- loads/runs the committed model.pt, no mAP claim (see that script's
     docstring for why).
  2. RAG retrieval evaluation (backend/app/rag_scripts/scripts/
     hybrid_retrieval.py --evaluate) -- the real, complete 50-incident P@3/
     MRR eval against the committed 185-chunk index.

Run from the repo root:
    python .github/scripts/generate_evaluation_results.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
RAG_SCRIPTS_DIR = BACKEND_DIR / "app" / "rag_scripts"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- best-effort metadata, never fatal
        return "unknown"


def run_yolo_smoke_test() -> dict:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / ".github" / "scripts" / "yolo_smoke_test.py")],
        cwd=str(BACKEND_DIR),
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("YOLO_RESULT_JSON:"):
            return json.loads(line[len("YOLO_RESULT_JSON:"):])
    return {
        "model_loaded": False,
        "errors": [f"yolo_smoke_test.py produced no YOLO_RESULT_JSON line (exit {completed.returncode})"],
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def run_rag_retrieval_eval() -> dict:
    completed = subprocess.run(
        [sys.executable, "scripts/hybrid_retrieval.py", "--evaluate"],
        cwd=str(RAG_SCRIPTS_DIR),
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "succeeded": False,
            "error": f"hybrid_retrieval.py --evaluate exited {completed.returncode}",
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    result: dict = {"succeeded": True}
    # Output format is "<Label>: {json block}" twice (Dense-only, Hybrid
    # RRF) -- see hybrid_retrieval.py's own main(). Parsed rather than
    # changed at the source, since that script is the reproducibility
    # fixture documented in backend/app/rag_scripts/README.md and changing
    # its output format isn't otherwise needed here.
    for label, key in (("Dense-only", "dense_only"), ("Hybrid RRF", "hybrid_rrf")):
        match = re.search(rf"{re.escape(label)}.*?:\s*(\{{.*?\}})", completed.stdout, re.DOTALL)
        if match:
            try:
                result[key] = json.loads(match.group(1))
            except json.JSONDecodeError:
                result[key] = {"parse_error": True, "raw": match.group(1)[:1000]}
        else:
            result[key] = {"not_found_in_output": True}
    result["raw_stdout"] = completed.stdout
    return result


def main() -> int:
    print("Running YOLO damage-detection smoke test...")
    yolo_result = run_yolo_smoke_test()
    print("Running RAG retrieval evaluation...")
    rag_result = run_rag_retrieval_eval()

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "yolo_damage_detection": yolo_result,
        "rag_retrieval_evaluation": rag_result,
        "yolo_map_not_computed_reason": (
            "Only 24 images / 6 label files are committed under "
            "data/vehide/images(labels)/test/ -- too small a sample for a "
            "meaningful mAP or confusion matrix. See "
            "docs/Milestone5_Report.md for the real validation/test-split "
            "numbers (from the full VehiDE dataset, not reproduced here)."
        ),
    }

    out_path = REPO_ROOT / "evaluation-results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    yolo_ok = bool(yolo_result.get("model_loaded")) and yolo_result.get("images_processed", 0) > 0
    rag_ok = bool(rag_result.get("succeeded"))
    print(f"YOLO smoke test: {'OK' if yolo_ok else 'FAILED'}")
    print(f"RAG retrieval evaluation: {'OK' if rag_ok else 'FAILED'}")
    return 0 if (yolo_ok and rag_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
