from __future__ import annotations

from typing import Any


class SeverityScoringService:
    def __init__(self, confidence_floor: float = 0.7):
        self.confidence_floor = confidence_floor

    def score_detections(self, detections: list[dict[str, Any]], image_width: int = 100, image_height: int = 100, image_area: int | None = None) -> dict[str, Any]:
        valid_detections = [
            detection for detection in detections
            if (detection.get("confidence") or 0.0) >= self.confidence_floor
        ]
        if not valid_detections and detections:
            valid_detections = list(detections)

        # image_area (the real photo's pixel area, from the caller) takes
        # precedence over image_width * image_height, which is only ever a
        # caller-supplied placeholder -- using it directly on a real photo's
        # bbox coordinates produces nonsensical ratios far above 1.0.
        if image_area and image_area > 0:
            effective_area = image_area
        elif image_width > 0 and image_height > 0:
            effective_area = image_width * image_height
        else:
            effective_area = 0

        area_sum = 0.0
        per_region: list[dict[str, Any]] = []
        for detection in valid_detections:
            area = self._area_from_detection(detection)
            area_sum += area
            per_region.append({
                "class_name": detection.get("class_name", "unknown"),
                "confidence": detection.get("confidence", 0.0),
                "severity": self._severity_label_for_ratio(area_sum / max(effective_area, 1)),
            })

        ratio = area_sum / effective_area if effective_area else 0.0

        overall_severity = self._severity_label_for_ratio(ratio)
        return {
            "overall_severity": overall_severity,
            "severity_score": round(ratio, 4),
            "per_region": per_region,
        }

    def _area_from_detection(self, detection: dict[str, Any]) -> float:
        polygon = detection.get("mask_polygon") or []
        if polygon:
            return self._polygon_area(polygon)

        bbox = detection.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            return width * height
        return 0.0

    def _polygon_area(self, polygon: list[list[float]] | None) -> float:
        if not polygon:
            return 0.0
        if len(polygon) < 3:
            return 0.0
        total = 0.0
        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            total += point[0] * next_point[1] - next_point[0] * point[1]
        return abs(total) / 2.0

    def _severity_label_for_ratio(self, ratio: float) -> str:
        if ratio >= 0.2:
            return "Severe"
        if ratio >= 0.08:
            return "Moderate"
        return "Minor"
