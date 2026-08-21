#!/usr/bin/env python3
"""Single entry point to reproduce this repository's key results.

Runs, in sequence, from a fresh clone:
  1. Install backend dependencies (backend/requirements.txt).
  2. Ensure a .env exists (copied from .env.example if missing) -- never
     overwrites an existing one, since it may already hold real secrets.
  3. Run the backend test suite (exercises the real 5-agent claim pipeline
     end to end: YOLO damage detection, severity scoring, RAG policy-clause
     retrieval, Groq report synthesis, deterministic fraud scoring).
  4. Run the no-API-key RAG retrieval evaluation (reproduces the P@3/MRR
     numbers in README.md's "Evaluation & Results" section against the committed
vector index).

What this script deliberately does NOT do (see README.md's "Model /
Pipeline Execution" and "Evaluation & Results" sections for why):
  - Retrain the YOLO model. That pipeline needs a GPU and a Kaggle account
    and is written for Google Colab -- it cannot run unattended on an
    arbitrary local machine. Run notebooks/Yolov11m_Training&
    HyperparameterTuning.ipynb directly instead.
  - Run the report-generation / RAGAs evaluations (backend/app/rag_scripts/
    scripts/eval_report_agent.py, ragas_eval.py). Both call a real LLM and
    need GROQ_API_KEY / GOOGLE_API_KEY -- this script checks for the keys
    and tells you the exact command to run yourself rather than silently
    skipping or spending your API quota without asking.
  - Install or run the frontend (npm). See README.md's "Running the
    Application" section.

Usage:
    python scripts/reproduce.py
    python scripts/reproduce.py --skip-install
    python scripts/reproduce.py --skip-tests --skip-rag-eval

Every path below is derived from this file's own location, never
hardcoded, so this script works from any clone location on any machine.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
RAG_SCRIPTS_DIR = BACKEND_DIR / "app" / "rag_scripts"
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _run(description: str, args: list[str], cwd: Path, env_note: str | None = None) -> bool:
    """Run a subprocess step, returning True on success (exit code 0)."""
    print(f"\n--- {description} ---")
    print(f"$ {' '.join(args)}  (cwd={cwd.relative_to(REPO_ROOT) if cwd != REPO_ROOT else '.'})")
    result = subprocess.run(args, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[FAILED] {description} (exit code {result.returncode})")
        if env_note:
            print(f"         {env_note}")
        return False
    print(f"[OK] {description}")
    return True


def ensure_env_file() -> None:
    _banner("Step: .env")
    if ENV_FILE.exists():
        print(f"[OK] {ENV_FILE.relative_to(REPO_ROOT)} already exists -- leaving it untouched.")
        return
    if not ENV_EXAMPLE.exists():
        print(f"[WARN] Neither .env nor .env.example found at repo root -- skipping.")
        return
    shutil.copy(ENV_EXAMPLE, ENV_FILE)
    print(f"[OK] Copied {ENV_EXAMPLE.name} -> .env. Edit it now and set GROQ_API_KEY "
          f"(and GOOGLE_API_KEY if you also want the RAGAs eval) before continuing.")


def has_env_var(name: str) -> bool:
    """Best-effort check of .env for a non-empty value, without executing
    any app code (so this works even before dependencies are installed)."""
    if not ENV_FILE.exists():
        return False
    for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return bool(value.strip().strip('"').strip("'"))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-install", action="store_true", help="Skip `pip install -r backend/requirements.txt`.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the backend pytest suite.")
    parser.add_argument("--skip-rag-eval", action="store_true", help="Skip the RAG retrieval evaluation.")
    args = parser.parse_args()

    results: dict[str, bool | None] = {}  # None = skipped

    _banner("Reproducing key results for the Car Damage Insurance Claim Portal")
    print(f"Repo root: {REPO_ROOT}")

    ensure_env_file()

    if args.skip_install:
        print("\n[SKIP] Backend dependency install (--skip-install)")
        results["Install backend dependencies"] = None
    else:
        _banner("Step: install backend dependencies")
        results["Install backend dependencies"] = _run(
            "pip install -r backend/requirements.txt",
            [sys.executable, "-m", "pip", "install", "-r", str(BACKEND_DIR / "requirements.txt")],
            cwd=REPO_ROOT,
        )

    if args.skip_tests:
        print("\n[SKIP] Backend test suite (--skip-tests)")
        results["Backend test suite (pytest)"] = None
    else:
        _banner("Step: backend test suite")
        results["Backend test suite (pytest)"] = _run(
            "python -m pytest -q",
            [sys.executable, "-m", "pytest", "-q"],
            cwd=BACKEND_DIR,
            env_note="A pre-existing failure here is a real regression -- expect 67 passed/2 skipped "
                     "without a live GROQ_API_KEY, or 69 passed/0 skipped with one.",
        )

    if args.skip_rag_eval:
        print("\n[SKIP] RAG retrieval evaluation (--skip-rag-eval)")
        results["RAG retrieval evaluation"] = None
    else:
        _banner("Step: RAG retrieval evaluation (reproduces the P@3/MRR numbers in README.md)")
        results["RAG retrieval evaluation"] = _run(
            "python scripts/hybrid_retrieval.py --evaluate",
            [sys.executable, "scripts/hybrid_retrieval.py", "--evaluate"],
            cwd=RAG_SCRIPTS_DIR,
            env_note="Needs chromadb/sentence-transformers from backend/requirements.txt (already "
                     "installed above). Note: the first run rewrites backend/app/rag_scripts/data/"
                     "chroma_db/* in place (chromadb's own index migration, not data loss) -- "
                     "`git status` will show it modified afterwards; see that folder's README.",
        )

    _banner("What this script did NOT run -- do these yourself")
    print("- Frontend: cd frontend && npm install && npm run dev   (see README.md, 'Running the Application')")
    print("- Start the backend: cd backend && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
    print("- YOLO retraining: notebooks/Yolov11m_Training&HyperparameterTuning.ipynb on Google Colab (see README.md, 'Retraining the damage-detection model')")

    groq_key = has_env_var("GROQ_API_KEY")
    google_key = has_env_var("GOOGLE_API_KEY")
    print(f"- Report-generation eval (needs GROQ_API_KEY{'' if groq_key else ', NOT currently set in .env'}):")
    print("    cd backend/app/rag_scripts && PYTHONPATH=. python scripts/eval_report_agent.py")
    print(f"- RAGAs LLM-judge eval (needs GROQ_API_KEY + GOOGLE_API_KEY{'' if groq_key and google_key else ', NOT currently both set in .env'}):")
    print("    cd backend/app/rag_scripts && PYTHONPATH=. python scripts/ragas_eval.py --all")

    _banner("Summary")
    ok = True
    for step, outcome in results.items():
        status = "SKIPPED" if outcome is None else ("OK" if outcome else "FAILED")
        print(f"  [{status:7}] {step}")
        if outcome is False:
            ok = False

    if not ok:
        print("\nOne or more steps failed -- see the [FAILED] section(s) above for details.")
        return 1
    print("\nAll steps that ran succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
