from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.services.langfuse_observability import LangfuseObserver
from app.services.mcp_tools import build_toolkit

logger = logging.getLogger("claims_portal.langgraph_orchestrator")

AGENT_DESCRIPTIONS = {
    "detect_damage": "Damage Detection agent: runs the YOLO model on the claim photos to find damage regions.",
    "score_severity": "Severity Scoring agent: computes a severity label from the damage regions already detected.",
    "retrieve_policy": "Policy Clause agent: retrieves coverage/exclusion clauses for the claim's detected damage classes.",
    "synthesize_report": "Report Synthesis agent: produces the structured claim assessment report from the other agents' outputs.",
    "assess_fraud": "Fraud agent: assesses fraud risk from severity, amount, and the synthesized report's confidence.",
}

COORDINATOR_SYSTEM_PROMPT = """You are the coordinator for an insurance-claim AI analysis pipeline.

At each turn you are given the set of agents that are currently valid to run \
next (their prerequisites are already satisfied) and a compact summary of \
what the pipeline has produced so far. Call exactly one of the tools you are \
given to choose which agent runs next, and give a one-sentence reason. \
You may only choose from the tools provided in this turn -- do not invent \
an agent name that isn't offered."""


class ClaimAnalysisState(TypedDict, total=False):
    claim: Any
    policy: Any
    detections: list[dict[str, Any]]
    image_area: int
    severity_summary: dict[str, Any]
    policy_findings: list[dict[str, Any]]
    report_json: dict[str, Any]
    fraud_assessment: dict[str, Any]
    needs_human_review: bool
    final_status: str
    planned_action: str
    completed_actions: list[str]
    trace: Any


class LangGraphClaimOrchestrator:
    def __init__(self, damage_service: Any, severity_service: Any, policy_service: Any, report_service: Any, fraud_service: Any | None = None, observer: LangfuseObserver | None = None):
        self.damage_service = damage_service
        self.severity_service = severity_service
        self.policy_service = policy_service
        self.report_service = report_service
        self.fraud_service = fraud_service
        self.observer = observer or LangfuseObserver()
        self.groq_api_key = getattr(report_service, "groq_api_key", None)
        self.groq_model = getattr(report_service, "groq_model", None)
        self.groq_base_url = getattr(report_service, "groq_base_url", "https://api.groq.com/openai/v1")
        self.toolkit = build_toolkit(
            damage_service=self.damage_service,
            severity_service=self.severity_service,
            policy_service=self.policy_service,
            report_service=self.report_service,
            fraud_service=self.fraud_service,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(ClaimAnalysisState)

        workflow.add_node("coordinator", self._coordinate)
        workflow.add_node("tool_execution", self._execute_tool)
        workflow.add_node("damage_detection", self._detect_damage)
        workflow.add_node("severity_scoring", self._score_severity)
        workflow.add_node("policy_clause_retrieval", self._retrieve_policy)
        workflow.add_node("report_synthesis", self._synthesize_report)
        workflow.add_node("fraud_assessment", self._assess_fraud)
        workflow.add_node("flag_human_review", self._flag_human_review)
        workflow.add_node("finalize_claim", self._finalize_claim)

        workflow.set_entry_point("coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "tool_execution": "tool_execution",
                "flag_human_review": "flag_human_review",
                "finalize_claim": "finalize_claim",
            },
        )
        workflow.add_edge("tool_execution", "coordinator")
        workflow.add_edge("flag_human_review", END)
        workflow.add_edge("finalize_claim", END)

        compiled = workflow.compile()
        return compiled

    def _coordinate(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        valid_actions = self._valid_next_actions(state)

        if not valid_actions:
            if self._should_escalate_to_human(state) == "escalate":
                state["planned_action"] = "flag_human_review"
            else:
                state["planned_action"] = "finalize_claim"
            return state

        if len(valid_actions) == 1:
            # No real choice to make -- skip the LLM call entirely.
            state["planned_action"] = valid_actions[0]
            return state

        state["planned_action"] = self._plan_next_action(state, valid_actions)
        return state

    def _valid_next_actions(self, state: ClaimAnalysisState) -> list[str]:
        """Agents whose prerequisites are already satisfied and that haven't
        run yet, in dependency order. Membership in completed_actions is the
        only gate (not truthiness of the produced data), so a step that
        legitimately produces an empty result -- e.g. detect_damage finding
        no detections -- is never re-triggered."""
        completed = set(state.get("completed_actions") or [])

        if "detect_damage" not in completed:
            return ["detect_damage"]

        # score_severity and retrieve_policy both only depend on detections
        # being available, so once detection is done either can run next --
        # this is the genuine choice point handed to the planner.
        parallel = [
            action for action in ("score_severity", "retrieve_policy")
            if action not in completed
        ]
        if parallel:
            return parallel

        if "synthesize_report" not in completed:
            return ["synthesize_report"]

        if self.fraud_service is not None and "assess_fraud" not in completed:
            return ["assess_fraud"]

        return []

    def _plan_next_action(self, state: ClaimAnalysisState, valid_actions: list[str]) -> str:
        trace = state.get("trace")
        context = self._planning_context(state)
        fallback = valid_actions[0]

        if not self.groq_api_key:
            self.observer.span(
                trace, name="coordinator_planning",
                input_data={"valid_actions": valid_actions, "context": context},
                output_data={"chosen_action": fallback, "reason": "GROQ_API_KEY not configured; used deterministic order"},
            )
            return fallback

        chosen, reason = fallback, None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.groq_api_key, base_url=self.groq_base_url)
            response = client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"valid_actions": valid_actions, "context": context})},
                ],
                tools=[self._tool_schema(action) for action in valid_actions],
                tool_choice="required",
                temperature=0,
            )
            tool_calls = response.choices[0].message.tool_calls or []
            if tool_calls:
                candidate = tool_calls[0].function.name
                try:
                    reason = json.loads(tool_calls[0].function.arguments or "{}").get("reason")
                except (json.JSONDecodeError, AttributeError):
                    reason = None
                if candidate in valid_actions:
                    chosen = candidate
                else:
                    reason = f"LLM chose unavailable action '{candidate}'; used deterministic fallback"
        except Exception as error:
            logger.warning("Coordinator planning call failed: %s", error)
            reason = f"Groq planning call failed: {error}"

        self.observer.span(
            trace, name="coordinator_planning",
            input_data={"valid_actions": valid_actions, "context": context},
            output_data={"chosen_action": chosen, "reason": reason},
        )
        return chosen

    def _planning_context(self, state: ClaimAnalysisState) -> dict[str, Any]:
        detections = state.get("detections") or []
        return {
            "completed_agents": state.get("completed_actions") or [],
            "detections_count": len(detections),
            "damage_classes": sorted({d.get("class_name") for d in detections if d.get("class_name")}),
            "has_severity_summary": bool(state.get("severity_summary")),
            "has_policy_findings": bool(state.get("policy_findings")),
            "has_report": bool(state.get("report_json")),
        }

    def _tool_schema(self, action: str) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": action,
                "description": AGENT_DESCRIPTIONS.get(action, action),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "One sentence explaining why this agent should run next."},
                    },
                    "required": ["reason"],
                },
            },
        }

    def _route_from_coordinator(self, state: ClaimAnalysisState) -> str:
        action = state.get("planned_action") or ""
        if action in {"flag_human_review", "finalize_claim"}:
            return action
        return "tool_execution"

    def _execute_tool(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        action = state.get("planned_action") or ""
        if action == "detect_damage":
            self._detect_damage(state)
        elif action == "score_severity":
            self._score_severity(state)
        elif action == "retrieve_policy":
            self._retrieve_policy(state)
        elif action == "synthesize_report":
            self._synthesize_report(state)
        elif action == "assess_fraud":
            self._assess_fraud(state)
        completed_actions = state.get("completed_actions") or []
        if action not in completed_actions:
            completed_actions.append(action)
        state["completed_actions"] = completed_actions
        return state

    def _detect_damage(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        claim = state.get("claim")
        photos = getattr(claim, "photos", []) or []
        detections: list[dict[str, Any]] = []
        total_image_area = 0
        get_dimensions = getattr(self.damage_service, "get_image_dimensions", None)
        for photo in photos:
            image_path = getattr(photo, "file_path", None)
            if image_path:
                detections.extend(self._call_tool("detect_damage_tool", image_path))
                if get_dimensions is not None:
                    width, height = get_dimensions(image_path)
                    total_image_area += (width or 0) * (height or 0)
        state["detections"] = detections
        state["image_area"] = total_image_area
        self.observer.span(
            state.get("trace"),
            name="damage_detection",
            input_data={"photo_paths": [getattr(p, "file_path", None) for p in photos]},
            output_data={"detections": detections, "image_area": total_image_area},
        )
        return state

    def _score_severity(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        detections = state.get("detections") or []
        image_area = state.get("image_area") or 0
        severity_summary = self._call_tool("score_severity_tool", detections, image_area=image_area)
        state["severity_summary"] = severity_summary
        self.observer.span(
            state.get("trace"),
            name="severity_scoring",
            input_data={"detections": detections},
            output_data=severity_summary,
        )
        return state

    def _retrieve_policy(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        claim = state.get("claim")
        policy = state.get("policy")
        detections = state.get("detections") or []
        damage_classes = sorted({d.get("class_name") for d in detections if d.get("class_name")})
        policy_input = {
            "policy_number": getattr(policy, "policy_number", None),
            "damage_classes": damage_classes,
            "incident_description": getattr(claim, "incident_description", ""),
            "claimed_amount": str(getattr(claim, "claimed_amount", None)),
            "policy_limit": str(getattr(policy, "policy_limit", None)),
        }
        policy_findings = self._call_tool(
            "retrieve_policy_clauses_tool",
            policy_input["policy_number"],
            damage_classes,
            policy_input["incident_description"],
            getattr(claim, "claimed_amount", None),
            getattr(policy, "policy_limit", None),
        )
        state["policy_findings"] = policy_findings
        self.observer.span(
            state.get("trace"),
            name="policy_clause_retrieval",
            input_data=policy_input,
            output_data={"policy_findings": policy_findings},
        )
        return state

    def _synthesize_report(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        detections = state.get("detections") or []
        severity_summary = state.get("severity_summary") or {}
        policy_findings = state.get("policy_findings") or []
        report_json = self._call_tool("synthesize_report_tool", detections, severity_summary, policy_findings)
        state["report_json"] = report_json
        self.observer.generation(
            state.get("trace"),
            name="report_synthesis",
            input_data={"detections": detections, "severity_summary": severity_summary, "policy_findings": policy_findings},
            output_data=report_json,
            model=getattr(self.report_service, "groq_model", None),
        )
        return state

    def _assess_fraud(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        if self.fraud_service is None:
            state["fraud_assessment"] = {"fraud_score": 0.0, "needs_investigation": False, "reason": "No fraud agent configured", "signals": []}
            return state
        claim = state.get("claim")
        policy = state.get("policy")
        severity_summary = state.get("severity_summary") or {}
        report_json = state.get("report_json") or {}

        already_claimed = Decimal("0")
        for sibling in getattr(policy, "claims", None) or []:
            if getattr(sibling, "claim_id", None) == getattr(claim, "claim_id", None):
                continue
            if getattr(sibling, "status", None) == "denied":
                continue
            already_claimed += getattr(sibling, "claimed_amount", None) or Decimal("0")

        expiry_date = getattr(policy, "expiry_date", None)
        policy_limit = getattr(policy, "policy_limit", None)
        fraud_input = {
            "claimant_name": getattr(claim, "claimant_name", ""),
            "claimed_amount": str(getattr(claim, "claimed_amount", None)),
            "incident_description": getattr(claim, "incident_description", ""),
            "severity_summary": severity_summary,
            "confidence_score": (report_json or {}).get("confidence_score", 0.5),
            "policy_holder_name": getattr(policy, "policy_holder_name", None),
            "policy_status": getattr(policy, "status", None),
            "incident_date": str(getattr(claim, "incident_date", "") or "") or None,
            "policy_expiry_date": str(expiry_date) if expiry_date else None,
            "already_claimed_amount": str(already_claimed),
            "policy_limit": str(policy_limit) if policy_limit is not None else None,
        }
        fraud_assessment = self._call_tool("assess_fraud_tool", **fraud_input)
        state["fraud_assessment"] = fraud_assessment
        self.observer.span(
            state.get("trace"),
            name="fraud_assessment",
            input_data=fraud_input,
            output_data=fraud_assessment,
        )
        return state

    def _should_escalate_to_human(self, state: ClaimAnalysisState) -> str:
        report_json = state.get("report_json") or {}
        confidence_score = report_json.get("confidence_score") or 0.0
        fraud_assessment = state.get("fraud_assessment") or {}
        if fraud_assessment.get("needs_investigation"):
            return "escalate"
        return "escalate" if confidence_score < 0.6 else "finalize"

    def _flag_human_review(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        state["needs_human_review"] = True
        state["final_status"] = "needs_human_review"
        return state

    def _finalize_claim(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        state["needs_human_review"] = False
        state["final_status"] = "finalized"
        return state

    def _call_tool(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        for tool in self.toolkit:
            if getattr(tool, "name", None) == tool_name:
                return tool.func(*args, **kwargs)
        return None

    def run(self, claim: Any, policy: Any = None) -> dict[str, Any]:
        trace = self.observer.start_trace(
            name="claim_analysis_pipeline",
            input_data={
                "claim_id": getattr(claim, "claim_id", None),
                "policy_number": getattr(policy, "policy_number", None),
                "incident_description": getattr(claim, "incident_description", None),
                "claimed_amount": str(getattr(claim, "claimed_amount", None)),
            },
            metadata={"agents": ["damage_detection", "severity_scoring", "policy_clause_retrieval", "report_synthesis", "fraud_assessment"]},
        )
        result = self.graph.invoke({"claim": claim, "policy": policy, "trace": trace})
        output = {
            "detections": result.get("detections", []),
            "severity_summary": result.get("severity_summary", {}),
            "policy_findings": result.get("policy_findings", []),
            "report_json": result.get("report_json", {}),
            "fraud_assessment": result.get("fraud_assessment", {}),
            "needs_human_review": result.get("needs_human_review", False),
        }
        self.observer.end_trace(trace, output_data=output)
        self.observer.flush()
        return output
