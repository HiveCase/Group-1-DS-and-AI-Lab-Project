"""
Provisional YOLO detection output schema for the Report Agent pipeline.

Why this exists: no YOLO model has been trained yet -- src/models/,
src/inference/, configs/model.yaml, and configs/inference.yaml are all still
empty placeholders, and docs/Milestone3_Report.md has no content past its
title page. The Report Agent context bundle (scripts/report_context.py) needs
*some* concrete detection format to build against now, so this defines one,
reusing constants already established elsewhere in this repo rather than
inventing new ones:

  - class_id / class names: configs/class_remap.json (project_class_names),
    matching the CLASS_ID mapping already used in scripts/eda_vehide.py.
  - severity bins: scripts/eda_vehide.py SEVERITY_BINS / SEVERITY_LABELS, the
    area-ratio proxy documented in Milestone 2 Section 5.3. That proxy is
    explicitly global (not per-class calibrated) -- Milestone 2 flags
    per-class calibration as still-needed future work, not something this
    project has done yet. Carried over here unchanged rather than inventing
    unvalidated per-class thresholds.

This schema (confidence, bbox_normalized as [x_center, y_center, w, h],
area_ratio) matches standard Ultralytics YOLO inference output. It must be
confirmed against whatever src/inference/ actually emits once the real model
exists -- treat every field here as provisional until then. Two things
called out explicitly as unresolved: (1) the training labels found in
data/vehide/labels/ are plain 5-field bounding boxes (class x y w h), not
variable-length segmentation polygons, despite "YOLO11m-seg" in Milestones
1-2 -- this schema assumes bbox output, not masks; (2) no confidence
threshold for the human-review escalation path is defined anywhere in the
repo ("a configurable threshold" is the only language used) -- the value
below is a placeholder, not a calibrated number.
"""
from dataclasses import dataclass, field
from math import sqrt

CLASS_ID = {
    "dent": 0,
    "scratch": 1,
    "crack": 2,
    "broken_lamp": 3,
    "shattered_glass": 4,
    "flat_tyre": 5,
}
CLASS_NAMES = list(CLASS_ID.keys())

# scripts/eda_vehide.py SEVERITY_BINS / SEVERITY_LABELS (global area-ratio proxy)
SEVERITY_BINS = [0.0, 0.02, 0.08, 1.0]
SEVERITY_LABELS = ["minor", "moderate", "severe"]

# Placeholder only -- no calibrated value exists anywhere in the repo.
ESCALATION_CONFIDENCE_THRESHOLD = 0.50


def severity_from_area_ratio(area_ratio: float) -> str:
    """Bucket a normalised bbox area ratio into the global Minor/Moderate/Severe
    proxy from scripts/eda_vehide.py (see module docstring for why this is
    global rather than per-class)."""
    for i in range(len(SEVERITY_BINS) - 1):
        if SEVERITY_BINS[i] <= area_ratio <= SEVERITY_BINS[i + 1]:
            return SEVERITY_LABELS[i]
    return SEVERITY_LABELS[-1]


@dataclass
class Detection:
    """One YOLO detection instance, in the provisional schema described above."""
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
    """Convenience constructor for synthetic/test detections: build a square
    bbox that yields the requested area_ratio directly, rather than requiring
    callers to work out width/height by hand."""
    side = sqrt(area_ratio)
    xc, yc = center
    return Detection(class_name=class_name, confidence=confidence,
                      bbox_normalized=[xc, yc, side, side])
