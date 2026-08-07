from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app.core.config import get_settings


class DamageDetectionService:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or self._resolve_default_model_path())
        self._model: Any | None = None

    def _resolve_default_model_path(self) -> Path:
        candidates = [
            Path("backend/model"),
            Path("backend/model/model.pt"),
            Path("backend/model/model.onnx"),
            Path("backend/models"),
            Path("backend/models/model.joblib"),
            Path("backend/models/model.pt"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("backend/model")

    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except Exception:
            return None

        model_path = self.model_path
        if model_path.is_dir():
            candidate_files = sorted(model_path.glob("*.pt")) + sorted(model_path.glob("*.onnx"))
            if candidate_files:
                model_path = candidate_files[0]
        if not model_path.exists():
            return None
        try:
            self._model = YOLO(str(model_path))
        except Exception:
            self._model = None
        return self._model

    def detect_from_path(self, image_path: str | Path) -> list[dict[str, Any]]:
        model = self._load_model()
        if model is None:
            return []

        try:
            predictions = model.predict(str(image_path), stream=False, imgsz=640)
        except TypeError:
            try:
                predictions = model.predict(str(image_path))
            except Exception:
                return []
        except Exception:
            return []

        detections: list[dict[str, Any]] = []
        for result in predictions:
            if isinstance(result, dict):
                raw_items = result.get("boxes") or result.get("detections") or []
                if not raw_items:
                    detection_keys = {"class_name", "bbox", "mask_polygon", "confidence"}
                    if detection_keys.intersection(result.keys()):
                        detections.append(self._normalize_detection(result))
                    continue
                for item in raw_items:
                    detections.append(self._normalize_detection(item))
                continue

            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for index, box in enumerate(boxes):
                cls = int(box.cls[index]) if hasattr(box, "cls") and len(box.cls) > index else 0
                conf = float(box.conf[index]) if hasattr(box, "conf") and len(box.conf) > index else 0.0
                xyxy = box.xyxy[index].tolist() if hasattr(box, "xyxy") and len(box.xyxy) > index else [0, 0, 0, 0]
                mask = getattr(box, "mask", None)
                polygon = None
                if mask is not None and hasattr(mask, "xy") and len(mask.xy) > 0:
                    polygon = mask.xy[0].tolist()
                detections.append({
                    "class_name": self._class_label(cls),
                    "bbox": [round(float(coord), 2) for coord in xyxy],
                    "mask_polygon": polygon,
                    "confidence": round(conf, 4),
                })
        return detections

    def _normalize_detection(self, item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            class_name = item.get("class_name") or self._class_label(int(item.get("class_id", 0)))
            bbox = item.get("bbox") or [0, 0, 0, 0]
            return {
                "class_name": class_name,
                "bbox": [round(float(coord), 2) for coord in bbox],
                "mask_polygon": item.get("mask_polygon"),
                "confidence": round(float(item.get("confidence", 0.0)), 4),
            }
        return {
            "class_name": getattr(item, "class_name", "unknown"),
            "bbox": [round(float(coord), 2) for coord in getattr(item, "bbox", [0, 0, 0, 0])],
            "mask_polygon": getattr(item, "mask_polygon", None),
            "confidence": round(float(getattr(item, "confidence", 0.0)), 4),
        }

    def _class_label(self, class_id: int) -> str:
        labels = {
            0: "dent",
            1: "scratch",
            2: "crack",
            3: "broken_lamp",
            4: "shattered_glass",
            5: "flat_tyre",
        }
        return labels.get(class_id, "unknown")

    def annotate_image(self, image_path: str | Path, detections: list[dict[str, Any]], output_path: str | Path | None = None) -> str | None:
        source = Path(image_path)
        destination = Path(output_path or source.with_suffix(".annotated.jpg"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            return None
        try:
            from PIL import Image, ImageDraw
        except Exception:
            shutil.copy2(source, destination)
            return str(destination)

        with Image.open(source) as image:
            draw = ImageDraw.Draw(image)
            for detection in detections:
                bbox = detection.get("bbox") or [0, 0, 0, 0]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            image.save(destination)
        return str(destination)

    def _default_model_dir(self) -> Path:
        settings = get_settings()
        return settings.model_dir
