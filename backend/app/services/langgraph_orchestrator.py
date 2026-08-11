from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.services.mcp_tools import build_toolkit


class ClaimAnalysisState(TypedDict, total=False):
    claim: Any
    policy: Any
    detections: list[dict[str, Any]]
    severity_summary: dict[str, Any]
    policy_findings: list[dict[str, Any]]
    report_json: dict[str, Any]
    fraud_assessment: dict[str, Any]
    needs_human_review: bool
    final_status: str
    planned_action: str
    completed_actions: list[str]


class LangGraphClaimOrchestrator:
    def __init__(self, damage_service: Any, severity_service: Any, policy_service: Any, report_service: Any, fraud_service: Any | None = None):
        self.damage_service = damage_service
        self.severity_service = severity_service
        self.policy_service = policy_service
        self.report_service = report_service
        self.fraud_service = fraud_service
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
        completed_actions = set(state.get("completed_actions") or [])

        if "detect_damage" not in completed_actions and not state.get("detections"):
            state["planned_action"] = "detect_damage"
            return state
        if "score_severity" not in completed_actions and not state.get("severity_summary"):
            state["planned_action"] = "score_severity"
            return state
        if "retrieve_policy" not in completed_actions and not state.get("policy_findings"):
            state["planned_action"] = "retrieve_policy"
            return state
        if "synthesize_report" not in completed_actions and not state.get("report_json"):
            state["planned_action"] = "synthesize_report"
            return state
        if self.fraud_service is not None and "assess_fraud" not in completed_actions and not state.get("fraud_assessment"):
            state["planned_action"] = "assess_fraud"
            return state
        if self._should_escalate_to_human(state) == "escalate":
            state["planned_action"] = "flag_human_review"
            return state
        state["planned_action"] = "finalize_claim"
        return state

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
        detections: list[dict[str, Any]] = []
        for photo in getattr(claim, "photos", []) or []:
            image_path = getattr(photo, "file_path", None)
            if image_path:
                detections.extend(self._call_tool("detect_damage_tool", image_path))
        state["detections"] = detections
        return state

    def _score_severity(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        detections = state.get("detections") or []
        state["severity_summary"] = self._call_tool("score_severity_tool", detections, image_width=100, image_height=100)
        return state

    def _retrieve_policy(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        claim = state.get("claim")
        policy = state.get("policy")
        detections = state.get("detections") or []
        damage_classes = sorted({d.get("class_name") for d in detections if d.get("class_name")})
        state["policy_findings"] = self._call_tool(
            "retrieve_policy_clauses_tool",
            getattr(policy, "policy_number", None),
            damage_classes,
            getattr(claim, "incident_description", ""),
            getattr(claim, "claimed_amount", None),
            getattr(policy, "policy_limit", None),
        )
        return state

    def _synthesize_report(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        detections = state.get("detections") or []
        severity_summary = state.get("severity_summary") or {}
        policy_findings = state.get("policy_findings") or []
        state["report_json"] = self._call_tool("synthesize_report_tool", detections, severity_summary, policy_findings)
        return state

    def _assess_fraud(self, state: ClaimAnalysisState) -> ClaimAnalysisState:
        if self.fraud_service is None:
            state["fraud_assessment"] = {"fraud_score": 0.0, "needs_investigation": False, "reason": "No fraud agent configured"}
            return state
        claim = state.get("claim")
        severity_summary = state.get("severity_summary") or {}
        report_json = state.get("report_json") or {}
        state["fraud_assessment"] = self._call_tool(
            "assess_fraud_tool",
            getattr(claim, "incident_description", ""),
            getattr(claim, "claimed_amount", None),
            severity_summary,
            (report_json or {}).get("confidence_score", 0.5),
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
        result = self.graph.invoke({"claim": claim, "policy": policy})
        return {
            "detections": result.get("detections", []),
            "severity_summary": result.get("severity_summary", {}),
            "policy_findings": result.get("policy_findings", []),
            "report_json": result.get("report_json", {}),
            "fraud_assessment": result.get("fraud_assessment", {}),
            "needs_human_review": result.get("needs_human_review", False),
        }
