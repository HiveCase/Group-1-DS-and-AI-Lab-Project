"""
Milestone 4 — YOLO training with named experiment presets.

Each experiment changes exactly one thing from the baseline so results are
attributable. Run on Kaggle (P100) or any CUDA machine:

    python scripts/train_yolo.py --experiment baseline
    python scripts/train_yolo.py --experiment cls_weight
    python scripts/train_yolo.py --experiment cosine_lr
    python scripts/train_yolo.py --experiment label_smooth
    python scripts/train_yolo.py --experiment small_imgsz     # contingency
    python scripts/train_yolo.py --list                       # show presets

All runs write to runs/training/<experiment>/ (results.csv, weights/best.pt,
plots). Compare with scripts/compare_experiments.py afterwards.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Baseline — every experiment below is baseline + exactly one change
# ---------------------------------------------------------------------------
BASELINE = dict(
    data="data/vehide_processed/damage.yaml",
    epochs=50,
    imgsz=1280,
    batch=4,                # P100 16GB at 1280px; raise to 8 at 960px
    optimizer="AdamW",
    lr0=0.001,              # 10x below scratch default — fine-tuning regime
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    cls=2.0,                # class loss gain (imbalance mitigation, M2 §8.2)
    patience=15,            # early stopping
    seed=42,
    deterministic=True,
    plots=True,
    save=True,
    project="runs/training",
)

EXPERIMENTS: dict[str, dict] = {
    "baseline": {},
    # Exp 2 — stronger class-loss gain for the 6.68:1 imbalance
    "cls_weight": dict(cls=3.0),
    # Exp 3 — cosine LR schedule with tighter final LR
    "cosine_lr": dict(cos_lr=True, lrf=0.001),
    # Exp 4 — label smoothing as regularisation
    "label_smooth": dict(label_smoothing=0.1),
    # Exp 5 (contingency) — smaller images, bigger batch, longer schedule.
    # Only run if minority-class F1 is still low after Exps 2-4.
    "small_imgsz": dict(imgsz=960, batch=8, epochs=75),
}


def build_args(experiment: str) -> dict:
    args = copy.deepcopy(BASELINE)
    args.update(EXPERIMENTS[experiment])
    args["name"] = experiment
    return args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="baseline")
    p.add_argument("--model", default="yolo11m-seg.pt",
                   help="Base weights (yolo11m-seg.pt or yolov8m-seg.pt)")
    p.add_argument("--data", default=None,
                   help="Override damage.yaml path (e.g. Kaggle input path)")
    p.add_argument("--list", action="store_true", help="List presets and exit")
    cli = p.parse_args()

    if cli.list:
        for name in sorted(EXPERIMENTS):
            delta = EXPERIMENTS[name] or {"(baseline)": ""}
            print(f"{name:14s} {json.dumps(delta)}")
        sys.exit(0)

    args = build_args(cli.experiment)
    if cli.data:
        args["data"] = cli.data

    print(f"=== Experiment: {cli.experiment} ===")
    print(json.dumps(args, indent=2))

    from ultralytics import YOLO
    model = YOLO(cli.model)
    model.train(**args)

    run_dir = Path(args["project"]) / args["name"]
    print(f"\nDone. Results: {run_dir}/results.csv")
    print(f"Best weights:  {run_dir}/weights/best.pt")
    print("Copy best weights into models/best.pt for the pipeline:")
    print(f"  cp {run_dir}/weights/best.pt models/best.pt")


if __name__ == "__main__":
    main()
