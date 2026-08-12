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
        settings = get_settings()
        candidates = [
            settings.model_dir,
            settings.model_dir / "model.pt",
            settings.model_dir / "model.onnx",
            settings.model_dir / "model.joblib"
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return settings.model_dir / "model.pt"

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
                        normalized = self._normalize_detection(result)
                        if self._is_valid_detection(normalized):
                            detections.append(normalized)
                    continue
                for item in raw_items:
                    normalized = self._normalize_detection(item)
                    if self._is_valid_detection(normalized):
                        detections.append(normalized)
                continue

            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            box_data = getattr(boxes, "data", None)
            if box_data is not None:
                for row in box_data:
                    if len(row) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls_id = row.tolist()
                    detection = {
                        "class_name": self._class_label(int(cls_id)),
                        "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                        "mask_polygon": None,
                        "confidence": round(float(conf), 4),
                    }
                    if self._is_valid_detection(detection):
                        detections.append(detection)
                continue

            for index, box in enumerate(boxes):
                cls = int(box.cls[index]) if hasattr(box, "cls") and len(box.cls) > index else 0
                conf = float(box.conf[index]) if hasattr(box, "conf") and len(box.conf) > index else 0.0
                xyxy = box.xyxy[index].tolist() if hasattr(box, "xyxy") and len(box.xyxy) > index else [0, 0, 0, 0]
                mask = getattr(box, "mask", None)
                polygon = None
                if mask is not None and hasattr(mask, "xy") and len(mask.xy) > 0:
                    polygon = mask.xy[0].tolist()
                detection = {
                    "class_name": self._class_label(cls),
                    "bbox": [round(float(coord), 2) for coord in xyxy],
                    "mask_polygon": polygon,
                    "confidence": round(conf, 4),
                }
                if self._is_valid_detection(detection):
                    detections.append(detection)
        return detections

    def get_image_dimensions(self, image_path: str | Path) -> tuple[int, int]:
        try:
            from PIL import Image
        except Exception:
            return (0, 0)
        try:
            with Image.open(image_path) as image:
                return image.size
        except Exception:
            return (0, 0)

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

    def _is_valid_detection(self, detection: dict[str, Any]) -> bool:
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            return False
        if detection.get("class_name", "unknown") == "unknown":
            return False
        x1, y1, x2, y2 = [float(coord) for coord in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return width > 0 and height > 0

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

    _CLASS_COLORS = {
        "dent": (37, 99, 235),
        "scratch": (6, 182, 212),
        "crack": (239, 68, 68),
        "broken_lamp": (16, 185, 129),
        "shattered_glass": (168, 85, 247),
        "flat_tyre": (236, 72, 153),
    }
    _DEFAULT_COLOR = (250, 204, 21)

    def annotate_image(self, image_path: str | Path, detections: list[dict[str, Any]], output_path: str | Path | None = None) -> str | None:
        source = Path(image_path)
        destination = Path(output_path or source.with_suffix(".annotated.jpg"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            return None
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            shutil.copy2(source, destination)
            return str(destination)

        with Image.open(source) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            # Line width and font size are scaled to the image's own
            # dimensions, not a fixed pixel count -- a fixed 2px box is
            # invisible once the dashboard shrinks a real (e.g. 640x640)
            # photo down to a small thumbnail via CSS.
            short_side = min(image.width, image.height)
            line_width = max(3, round(short_side * 0.008))
            font_size = max(16, round(short_side * 0.045))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            for detection in detections:
                bbox = detection.get("bbox") or [0, 0, 0, 0]
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
                class_name = detection.get("class_name", "unknown")
                confidence = detection.get("confidence", 0.0) or 0.0
                color = self._CLASS_COLORS.get(class_name, self._DEFAULT_COLOR)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

                label = f"{class_name} {confidence:.2f}"
                text_box = draw.textbbox((0, 0), label, font=font)
                text_width = text_box[2] - text_box[0]
                text_height = text_box[3] - text_box[1]
                label_x1, label_y1 = x1, max(0, y1 - text_height - 10)
                label_x2, label_y2 = label_x1 + text_width + 10, label_y1 + text_height + 10
                draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)
                draw.text((label_x1 + 5, label_y1 + 3), label, fill=(0, 0, 0), font=font)

            image.save(destination, format="JPEG", quality=90)
        return str(destination)

    def _default_model_dir(self) -> Path:
        settings = get_settings()
        return settings.model_dir
