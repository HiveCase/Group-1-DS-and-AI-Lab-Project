"""
Typed data shapes for the claim pipeline: what a detection is, and what the
LangGraph state carries between nodes.

ClaimState is a TypedDict rather than a pydantic model because LangGraph
checkpoints it by serialising the state dict directly -- keeping it plain
means what is stored in the SQLite checkpoint is exactly what the nodes
read, with no validation layer that could accept a resumed state a node
then can't use. Detection stays a dataclass (converted to/from plain dicts
at the state boundary) so the derived-field logic below has one home.
"""
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Literal, Optional, TypedDict

from src.config import CLASS_ID, CLASS_NAMES, SEVERITY_BINS, SEVERITY_LABELS

ClaimStatus = Literal["running", "awaiting_review", "completed", "failed"]


def severity_from_area_ratio(area_ratio: float) -> str:
    """Bucket a normalised bbox area ratio into the global minor/moderate/
    severe proxy. Global rather than per-class: Milestone 2 flagged per-class
    calibration as still-needed future work, so inventing per-class
    thresholds here would present untuned numbers as tuned."""
    for i in range(len(SEVERITY_BINS) - 1):
        if SEVERITY_BINS[i] <= area_ratio <= SEVERITY_BINS[i + 1]:
            return SEVERITY_LABELS[i]
    return SEVERITY_LABELS[-1]


@dataclass
class Detection:
    """One damage instance detected on the claim photo.

    area_ratio is derived from the *bounding box* (w * h), even when the
    detecting model is a segmentation model that could give a tighter mask
    area -- see src/inference/yolo_detector.py for why that choice is
    load-bearing rather than incidental.
    """
    class_name: str
    confidence: float
    bbox_normalized: list  # [x_center, y_center, width, height], each in [0, 1]

    area_ratio: float = field(init=False)
    severity: str = field(init=False)
    class_id: int = field(init=False)

    def __post_init__(self):
        if self.class_name not in CLASS_ID:
            raise ValueError(f"Unknown class_name '{self.class_name}', expected one of {CLASS_NAMES}")
        self.class_id = CLASS_ID[self.class_name]
        w, h = self.bbox_normalized[2], self.bbox_normalized[3]
        self.area_ratio = round(w * h, 5)
        self.severity = severity_from_area_ratio(self.area_ratio)

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox_normalized": [round(v, 4) for v in self.bbox_normalized],
            "area_ratio": self.area_ratio,
            "severity": self.severity,
        }


def make_detection(class_name: str, area_ratio: float, confidence: float,
                   center: tuple = (0.5, 0.5)) -> Detection:
    """Build a Detection that lands on a requested area_ratio directly, by
    constructing a square bbox of the right size. For tests and fixtures --
    real detections come from src/inference/yolo_detector.py."""
    side = sqrt(area_ratio)
    xc, yc = center
    return Detection(class_name=class_name, confidence=confidence,
                     bbox_normalized=[xc, yc, side, side])


class ClaimState(TypedDict, total=False):
    """State threaded through the claim graph. `total=False` because nodes
    populate it progressively -- a state checkpointed mid-run (e.g. paused at
    human review) legitimately has no `final_report` yet."""

    # Inputs, set at submission
    claim_id: str
    user_id: str
    image_path: str
    incident_narrative: str

    # detect_damage
    detections: list  # list[dict] -- Detection.to_dict()

    # retrieve_clauses
    clauses_by_class: dict
    missing_coverage_for: list

    # build_context_bundle
    context_bundle: dict

    # generate_report
    report: Optional[dict]
    report_model: str
    report_error: Optional[str]

    # check_faithfulness
    faithfulness: dict

    # Escalation / routing
    needs_human_review: bool
    escalation_reasons: list

    # human_review (populated on resume)
    review_decision: Optional[dict]

    # finalize
    status: ClaimStatus
    final_report: Optional[dict]
    errors: list


def new_claim_state(claim_id: str, user_id: str, image_path: str,
                    incident_narrative: str) -> ClaimState:
    return ClaimState(
        claim_id=claim_id,
        user_id=user_id,
        image_path=str(image_path),
        incident_narrative=incident_narrative,
        detections=[],
        clauses_by_class={},
        missing_coverage_for=[],
        context_bundle={},
        report=None,
        report_model="",
        report_error=None,
        faithfulness={},
        needs_human_review=False,
        escalation_reasons=[],
        review_decision=None,
        status="running",
        final_report=None,
        errors=[],
    )
