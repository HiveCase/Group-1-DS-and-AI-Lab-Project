"""
The claim-processing graph: wiring, shared resources, and the run/resume API.

    detect_damage -> retrieve_clauses -> build_bundle -> generate_report
        -> check_faithfulness -> (needs review?) -> human_review --resume--> finalize
                                                  \\-> finalize

Threading model: one thread per claim, id "{user_id}:{claim_id}". The
persistent, cross-claim thing in this system is the user's ingested policy
index (src/retrieval/policy_store.py), not graph state -- a user uploads
once and files many claims, each an independent run against that index. A
single ever-growing per-user thread was the alternative and was rejected:
it would accumulate every claim a user ever filed into one state object,
making each run's state harder to reason about and to resume, for no gain.

Checkpointing uses SqliteSaver against a file in data/claims/. This is what
makes the human-review pause real rather than cosmetic: a claim halted at
human_review survives process exit, and resuming continues from that node
without re-running detection, retrieval, or the LLM call.
"""
import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.config import CHECKPOINT_DB, YOLO_CHECKPOINT
from src.schemas import ClaimState

log = logging.getLogger(__name__)


class Resources:
    """Expensive, reusable objects shared across claims in one process.

    Everything here is lazy: a run that never reaches the LLM never builds a
    client, and the CLI's read-only commands never load the 24MB checkpoint.
    Retrievers are cached per user because a user's TF-IDF matrix is rebuilt
    from their chunk TSV on construction.
    """

    def __init__(self, yolo_checkpoint=YOLO_CHECKPOINT):
        self.yolo_checkpoint = yolo_checkpoint
        self._detector = None
        self._llm_client = None
        self._clause_retrievers = {}

    @property
    def detector(self):
        if self._detector is None:
            from src.inference.yolo_detector import DamageDetector
            self._detector = DamageDetector(self.yolo_checkpoint)
        return self._detector

    @property
    def llm_client(self):
        if self._llm_client is None:
            from src.agents.report_agent import get_client
            self._llm_client = get_client()
        return self._llm_client

    def get_clause_retriever(self, user_id: str):
        if user_id not in self._clause_retrievers:
            from src.retrieval.clause_retriever import ClauseRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever
            self._clause_retrievers[user_id] = ClauseRetriever(HybridRetriever(user_id))
        return self._clause_retrievers[user_id]


def thread_id(user_id: str, claim_id: str) -> str:
    return f"{user_id}:{claim_id}"


def thread_config(user_id: str, claim_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id(user_id, claim_id)}}


def build_graph(resources: Resources):
    """Wire the nodes. Returns an uncompiled StateGraph so callers choose
    their own checkpointer (tests use an in-memory one)."""
    from src.agents import nodes

    def _bind(fn):
        # Nodes take (state, resources); LangGraph calls them with (state).
        def node(state: ClaimState):
            return fn(state, resources)
        node.__name__ = fn.__name__
        return node

    g = StateGraph(ClaimState)
    g.add_node("detect_damage", _bind(nodes.detect_damage))
    g.add_node("retrieve_clauses", _bind(nodes.retrieve_clauses))
    g.add_node("build_bundle", _bind(nodes.build_bundle))
    g.add_node("generate_report", _bind(nodes.generate_report))
    g.add_node("check_faithfulness", _bind(nodes.check_faithfulness))
    g.add_node("human_review", _bind(nodes.human_review))
    g.add_node("finalize", _bind(nodes.finalize))

    g.add_edge(START, "detect_damage")
    g.add_edge("detect_damage", "retrieve_clauses")
    g.add_edge("retrieve_clauses", "build_bundle")
    g.add_edge("build_bundle", "generate_report")
    g.add_edge("generate_report", "check_faithfulness")
    g.add_conditional_edges(
        "check_faithfulness",
        nodes.route_after_faithfulness,
        {"human_review": "human_review", "finalize": "finalize"},
    )
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g


def _checkpointer():
    """SqliteSaver over a file, with the connection left open for the
    process lifetime.

    SqliteSaver.from_conn_string is a context manager that closes the
    connection on exit, which is wrong for a long-lived CLI/service object;
    the connection is built directly instead. check_same_thread=False
    because LangGraph may touch it from a worker thread.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    return SqliteSaver(conn)


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Compiled graph + shared resources, built once per process."""
    resources = Resources()
    graph = build_graph(resources).compile(checkpointer=_checkpointer())
    return graph, resources


def submit_claim(claim_id: str, user_id: str, image_path: str,
                 incident_narrative: str) -> dict:
    """Run a new claim. Returns the final state, which may be paused at
    human review rather than complete -- check state["status"]."""
    from src.retrieval.policy_store import has_policy
    from src.schemas import new_claim_state

    if ":" in user_id or ":" in claim_id:
        # Thread ids are "{user_id}:{claim_id}" and are split on the first
        # colon to recover both. A colon in either half would make that
        # split silently attribute the claim to the wrong user.
        raise ValueError("user_id and claim_id must not contain ':'")

    if not has_policy(user_id):
        raise FileNotFoundError(
            f"User '{user_id}' has no ingested policy. Run: "
            f"python -m src.cli ingest-policy --user-id {user_id} --pdf <policy.pdf>"
        )

    graph, _ = get_compiled_graph()
    config = thread_config(user_id, claim_id)

    # Refuse to re-run a claim id that already exists. Without this, a second
    # submit under the same id silently replaces the first assessment: the
    # graph re-runs from the start on the same thread and the original
    # verdict is simply gone. For a claims system that is an audit-trail
    # hole -- a claim could be resubmitted with a reworded narrative until it
    # produced a favourable verdict, leaving no record of the earlier ones.
    if graph.get_state(config).created_at:
        raise ValueError(
            f"Claim '{claim_id}' already exists for user '{user_id}'. "
            "Claim ids are immutable; submit under a new id."
        )

    initial = new_claim_state(claim_id, user_id, image_path, incident_narrative)
    graph.invoke(initial, config=config)
    return get_claim_state(user_id, claim_id)


def resume_claim(user_id: str, claim_id: str, decision: dict) -> dict:
    """Resume a claim paused at human review, with the reviewer's decision.

    Execution continues from inside human_review -- detection, retrieval and
    the LLM call do not re-run.
    """
    from langgraph.types import Command

    from src.config import VALID_REVIEW_DECISIONS

    # Validate here, not only in the CLI's argparse: an unrecognised verb
    # used to fall through finalize's dispatch and be recorded as
    # "auto_assessed" on a claim whose own record said human_reviewed=True --
    # an audit entry that contradicted itself about who decided the claim.
    if isinstance(decision, str):
        decision = {"decision": decision}
    if not isinstance(decision, dict):
        raise ValueError(f"Review decision must be a dict or string, got {type(decision).__name__}")
    verb = (decision.get("decision") or "").lower()
    if verb not in VALID_REVIEW_DECISIONS:
        raise ValueError(
            f"Invalid review decision {decision.get('decision')!r}. "
            f"Expected one of {sorted(VALID_REVIEW_DECISIONS)}."
        )
    if verb == "override" and not decision.get("overridden_report"):
        raise ValueError("decision='override' requires an 'overridden_report'")
    decision["decision"] = verb

    graph, _ = get_compiled_graph()
    config = thread_config(user_id, claim_id)

    snapshot = graph.get_state(config)
    if not snapshot.created_at:
        raise ValueError(f"No claim found for thread '{thread_id(user_id, claim_id)}'")
    if not snapshot.next:
        raise ValueError(
            f"Claim '{claim_id}' is not awaiting review (it has already finished).")

    graph.invoke(Command(resume=decision), config=config)
    return get_claim_state(user_id, claim_id)


def get_claim_state(user_id: str, claim_id: str) -> dict:
    """Current checkpointed state of a claim, with a derived status.

    status is derived from the graph snapshot rather than trusted from the
    stored value: a run paused at an interrupt never executed finalize, so
    its stored status is still whatever the last completed node left.
    """
    graph, _ = get_compiled_graph()
    snapshot = graph.get_state(thread_config(user_id, claim_id))
    if not snapshot.created_at:
        raise ValueError(f"No claim found for thread '{thread_id(user_id, claim_id)}'")

    state = dict(snapshot.values)
    state["status"] = "awaiting_review" if snapshot.next else state.get("status", "completed")
    state["_next_nodes"] = list(snapshot.next)
    return state


def list_claims(user_id: str = None) -> list:
    """Every checkpointed claim. Optionally filtered to a user.

    Reads the checkpointer directly -- claims are not tracked anywhere else,
    so the checkpoint store is the register of what exists.

    The thread ids are collected from the checkpointer cursor and that cursor
    is fully consumed BEFORE any get_state() call. SqliteSaver holds a single
    connection, and issuing a query against it while its own list() cursor is
    still open deadlocks -- which is exactly what an earlier version of this
    function did, hanging indefinitely instead of returning.
    """
    graph, _ = get_compiled_graph()

    thread_ids = []
    for snapshot in graph.checkpointer.list(None):
        tid = snapshot.config["configurable"]["thread_id"]
        if ":" in tid and tid not in thread_ids:
            thread_ids.append(tid)

    claims = []
    for tid in thread_ids:
        uid, cid = tid.split(":", 1)
        if user_id and uid != user_id:
            continue
        snapshot = graph.get_state({"configurable": {"thread_id": tid}})
        values = snapshot.values
        claims.append({
            "thread_id": tid,
            "user_id": uid,
            "claim_id": cid,
            "status": "awaiting_review" if snapshot.next else values.get("status", "unknown"),
            "escalation_reasons": values.get("escalation_reasons", []),
        })
    return claims
