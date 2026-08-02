"""
Command-line entry point for the claim pipeline.

    python -m src.cli ingest-policy --user-id alice --pdf policy.pdf
    python -m src.cli submit-claim  --user-id alice --claim-id CLAIM_01 \
        --image photo.jpg --narrative "Rear-ended at a light."
    python -m src.cli list-pending
    python -m src.cli show-claim    --user-id alice --claim-id CLAIM_01
    python -m src.cli resume-claim  --user-id alice --claim-id CLAIM_01 \
        --decision approve --reviewer kamal

This is also the reviewer surface. There is no reviewer UI: `list-pending`
shows what is waiting, `show-claim` shows why, and `resume-claim` releases
it. That keeps the human-in-the-loop path exercisable end to end without
committing to an interface before the pipeline it would wrap is settled.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import CLAIM_OUTPUT_DIR


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )


def _save_final(state: dict):
    """Persist the terminal record next to the checkpoint store, for claims
    that finished. Paused claims stay only in the checkpoint -- writing a
    partial 'final' report would misrepresent an unfinished claim."""
    if not state.get("final_report"):
        return None
    CLAIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = CLAIM_OUTPUT_DIR / f"{state['user_id']}__{state['claim_id']}.json"
    with open(path, "w") as f:
        json.dump(state["final_report"], f, indent=2)
    return path


def _print_outcome(state: dict):
    status = state.get("status")
    print(f"\nclaim:   {state['claim_id']}  (user: {state['user_id']})")
    print(f"status:  {status}")

    detections = state.get("detections", [])
    print(f"damage:  {len(detections)} detection(s)")
    for d in detections:
        print(f"           - {d['class_name']:16s} severity={d['severity']:8s} conf={d['confidence']}")

    if status == "awaiting_review":
        print("\nawaiting human review:")
        for r in state.get("escalation_reasons", []):
            print(f"  ! {r}")
        print(f"\nresume with:\n  python -m src.cli resume-claim --user-id {state['user_id']} "
              f"--claim-id {state['claim_id']} --decision approve --reviewer <you>")
        return

    report = (state.get("final_report") or {}).get("report") or state.get("report")
    if report:
        print(f"\nverdicts ({state.get('report_model')}):")
        for item in report.get("items", []):
            print(f"  - {item.get('damage_class'):16s} {item.get('verdict'):12s} "
                  f"cites={item.get('cited_chunk_ids')}")
        print(f"\nrecommendation: {report.get('overall_recommendation')}")

    f = state.get("faithfulness", {})
    if f:
        print(f"\ngrounding: passed={f.get('passed')} "
              f"citations={f.get('valid_citations')}/{f.get('total_citations')} valid")
        for hf in f.get("hard_failures", []):
            print(f"  FAIL {hf}")
        for sf in f.get("soft_flags", []):
            print(f"  flag {sf}")

    outcome = (state.get("final_report") or {}).get("outcome")
    if outcome:
        print(f"\noutcome: {outcome}")


def cmd_ingest_policy(args):
    from src.retrieval.policy_store import ingest_policy

    manifest = ingest_policy(args.pdf, args.user_id)
    print(json.dumps(manifest.to_dict(), indent=2))


def cmd_submit_claim(args):
    from src.agents.graph import submit_claim

    state = submit_claim(
        claim_id=args.claim_id,
        user_id=args.user_id,
        image_path=args.image,
        incident_narrative=args.narrative,
    )
    _print_outcome(state)
    path = _save_final(state)
    if path:
        print(f"\nsaved -> {path}")


def cmd_list_pending(args):
    from src.agents.graph import list_claims

    claims = list_claims(args.user_id)
    pending = [c for c in claims if c["status"] == "awaiting_review"]

    if not args.all and not pending:
        print("No claims awaiting review.")
        return

    shown = claims if args.all else pending
    print(f"{'claim':34s} {'user':16s} {'status':16s} reasons")
    for c in shown:
        reasons = "; ".join(c["escalation_reasons"])[:60]
        print(f"{c['claim_id']:34s} {c['user_id']:16s} {c['status']:16s} {reasons}")


def cmd_show_claim(args):
    from src.agents.graph import get_claim_state

    state = get_claim_state(args.user_id, args.claim_id)
    if args.json:
        # _next_nodes is a live graph detail, not part of the claim record.
        print(json.dumps({k: v for k, v in state.items() if k != "_next_nodes"},
                         indent=2, default=str))
        return
    _print_outcome(state)


def cmd_resume_claim(args):
    from src.agents.graph import resume_claim

    decision = {"decision": args.decision, "reviewer": args.reviewer, "notes": args.notes}
    if args.decision == "override":
        if not args.report_file:
            sys.exit("--report-file is required with --decision override")
        with open(args.report_file) as f:
            decision["overridden_report"] = json.load(f)

    state = resume_claim(args.user_id, args.claim_id, decision)
    _print_outcome(state)
    path = _save_final(state)
    if path:
        print(f"\nsaved -> {path}")


def cmd_list_users(args):
    from src.retrieval.policy_store import list_users, load_manifest

    users = list_users()
    if not users:
        print("No users have ingested a policy yet.")
        return
    print(f"{'user':20s} {'chunks':>7s}  {'ingested':20s} source")
    for uid in users:
        m = load_manifest(uid)
        print(f"{uid:20s} {m.n_chunks:>7d}  {m.ingested_at:20s} {m.source_pdf}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="src.cli", description="Motor-insurance claim assessment pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="log pipeline progress")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest-policy", help="Ingest a user's uploaded policy PDF")
    p.add_argument("--user-id", required=True)
    p.add_argument("--pdf", required=True, type=Path)
    p.set_defaults(func=cmd_ingest_policy)

    p = sub.add_parser("submit-claim", help="Run a new claim through the pipeline")
    p.add_argument("--user-id", required=True)
    p.add_argument("--claim-id", required=True)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--narrative", required=True)
    p.set_defaults(func=cmd_submit_claim)

    p = sub.add_parser("list-pending", help="Claims awaiting human review")
    p.add_argument("--user-id", default=None)
    p.add_argument("--all", action="store_true", help="include finished claims")
    p.set_defaults(func=cmd_list_pending)

    p = sub.add_parser("show-claim", help="Show a claim's current state")
    p.add_argument("--user-id", required=True)
    p.add_argument("--claim-id", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show_claim)

    p = sub.add_parser("resume-claim", help="Release a claim paused for review")
    p.add_argument("--user-id", required=True)
    p.add_argument("--claim-id", required=True)
    p.add_argument("--decision", required=True, choices=["approve", "reject", "override"])
    p.add_argument("--reviewer", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--report-file", type=Path, default=None,
                   help="Corrected report JSON (required with --decision override)")
    p.set_defaults(func=cmd_resume_claim)

    p = sub.add_parser("list-users", help="Users with an ingested policy")
    p.set_defaults(func=cmd_list_users)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    # Expected operator errors (no policy ingested, claim not paused, unknown
    # claim, missing file) carry actionable messages already -- surface those
    # rather than a traceback. Anything else is a real bug and keeps its
    # traceback, which is the useful output for it.
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
