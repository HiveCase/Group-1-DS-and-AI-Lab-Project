"""
Runs the trained damage-segmentation model on a claim photo and returns
Detection objects.

This closes the gap the scripts/ pipeline never closed: scripts/yolo_schema.py
defined a *provisional* detection schema and said outright that it "must be
confirmed against whatever src/inference/ actually emits once the real model
exists" -- until now, nothing produced real detections, and every downstream
consumer built them synthetically via make_detection(). This module is that
confirmation.

Two things the real checkpoint settles that were open questions in
scripts/yolo_schema.py:

1. It IS a segmentation model (task="segment"), not the bbox-only detector
   that schema assumed, and its class names/order match CLASS_ID exactly
   (dent, scratch, crack, broken_lamp, shattered_glass, flat_tyre). So the
   class mapping needs no remapping layer.

2. Despite masks being available, area_ratio is deliberately computed from
   the BOUNDING BOX (w * h), not the mask. The severity bins
   ([0, 0.02, 0.08, 1.0], src/config.py) were fit in scripts/eda_vehide.py
   against normalised *bbox* area, derived from the dataset's polygons via
   _polygon_to_bbox. A segmentation mask's area is always <= its bbox area,
   and dramatically smaller for exactly the thin, elongated damage this
   dataset is full of -- a long scratch's mask might cover a few percent of
   its bbox. Switching to mask area would therefore push a large fraction of
   detections down a severity bucket while appearing to be a mere precision
   improvement. Using mask area would require re-fitting the bins against
   mask areas first; that is a real potential refinement, but it is a
   recalibration task, not a drop-in change, so the calibrated quantity is
   what gets computed here.
"""
import logging
from pathlib import Path

from src.config import (
    CLASS_NAMES,
    YOLO_CHECKPOINT,
    YOLO_CONF_THRESHOLD,
    YOLO_IMG_SIZE,
)
from src.schemas import Detection

log = logging.getLogger(__name__)


class DamageDetector:
    """Wraps the Ultralytics model. Loading a checkpoint costs seconds, so
    callers that process more than one claim should construct this once and
    reuse it (the graph builder in src/agents/graph.py does)."""

    def __init__(self, checkpoint: Path = YOLO_CHECKPOINT,
                 conf_threshold: float = YOLO_CONF_THRESHOLD,
                 img_size: int = YOLO_IMG_SIZE):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"YOLO checkpoint not found at {checkpoint}. "
                "Train one (scripts/train_yolo.py) or point YOLO_CHECKPOINT at an existing .pt"
            )
        # Imported lazily: ultralytics pulls in torch, which is a multi-second
        # import. Nothing that only does retrieval should pay for it.
        from ultralytics import YOLO

        self.checkpoint = checkpoint
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.model = YOLO(str(checkpoint))

        model_classes = set(self.model.names.values())
        unexpected = model_classes - set(CLASS_NAMES)
        if unexpected:
            raise ValueError(
                f"Checkpoint {checkpoint.name} emits classes not in the project schema: "
                f"{sorted(unexpected)}. Expected a subset of {CLASS_NAMES}."
            )

    def detect(self, image_path: str) -> list:
        """Run inference on one image, returning a list of Detection.

        An empty list is a legitimate result (the model found no damage it
        was confident about), not an error -- the caller decides what that
        means for the claim. src/agents/nodes.py escalates it for human
        review rather than reporting "nothing is covered".
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Claim image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            imgsz=self.img_size,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            # xywhn == [x_center, y_center, w, h] already normalised to the
            # image, which is exactly the Detection schema's bbox_normalized.
            for box in result.boxes:
                class_name = self.model.names[int(box.cls.item())]
                xywhn = [float(v) for v in box.xywhn[0].tolist()]
                detections.append(Detection(
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox_normalized=xywhn,
                ))

        log.info("Detected %d damage instance(s) in %s", len(detections), image_path.name)
        return detections
