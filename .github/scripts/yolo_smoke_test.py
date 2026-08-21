"""CI smoke test for the committed YOLO damage-detection model.

Deliberately does NOT compute or report mAP / a confusion matrix. Only
data/vehide/images/test/ is committed to this repo (24 images, 6 with a
matching label file) -- far too small a sample for a meaningful detection
metric, and reporting one from it would repeat the exact kind of
small-sample overclaim flagged in review feedback on this project (see
docs/Milestone5_Report.md's own n=5 robustness-eval caveat for the same
concern in a different evaluation). The real mAP / confusion-matrix
numbers are the validation-split and test-split figures already reported
in docs/Milestone5_Report.md and docs/Comprehensive_Technical_Documentation.md,
generated from the full VehiDE splits -- this script only confirms the
committed model.pt still loads and produces predictions, run on demand
rather than as a merge gate.

Run from the `backend/` directory (needs `app.*` importable, i.e. the same
PYTHONPATH setup backend/pytest.ini already provides):
    cd backend && python ../.github/scripts/yolo_smoke_test.py
Prints one line prefixed "YOLO_RESULT_JSON:" for the caller to extract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_IMAGES_DIR = REPO_ROOT / "data" / "vehide" / "images" / "test"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.services.damage_detection_service import DamageDetectionService

    result: dict = {
        "model_path": None,
        "model_loaded": False,
        "images_dir": str(TEST_IMAGES_DIR.relative_to(REPO_ROOT)),
        "images_found": 0,
        "images_processed": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_by_class": {},
        "errors": [],
        "note": (
            "Smoke test only -- confirms the committed model loads and runs "
            "inference. No mAP/confusion-matrix is computed here; see "
            "docs/Milestone5_Report.md for those numbers (see this script's "
            "own docstring for why)."
        ),
    }

    service = DamageDetectionService()
    result["model_path"] = str(service.model_path.resolve())

    images = sorted(TEST_IMAGES_DIR.glob("*.jpg")) if TEST_IMAGES_DIR.exists() else []
    result["images_found"] = len(images)
    if not images:
        result["errors"].append(f"No .jpg images found under {TEST_IMAGES_DIR}")
        print("YOLO_RESULT_JSON:" + json.dumps(result))
        return 1

    for image_path in images:
        try:
            detections = service.detect_from_path(image_path)
        except Exception as error:  # noqa: BLE001 -- surfaced in results, not fatal to the job
            result["errors"].append(f"{image_path.name}: {error}")
            continue
        result["images_processed"] += 1
        if detections:
            result["images_with_detections"] += 1
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            result["detections_by_class"][class_name] = result["detections_by_class"].get(class_name, 0) + 1
            result["total_detections"] += 1

    result["model_loaded"] = result["images_processed"] > 0 or not result["errors"]
    print("YOLO_RESULT_JSON:" + json.dumps(result))
    return 0 if result["images_processed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
