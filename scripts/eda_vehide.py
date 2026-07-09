import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
# Vietnamese class name → project class name
VN_TO_PROJECT = {
    "tray_son":    "scratch",
    "rach":        "scratch",
    "mop_lom":     "dent",
    "be_den":      "broken_lamp",
    "thung":       "flat_tyre",
    "vo_kinh":     "shattered_glass",
    "mat_bo_phan": "exclude",
}

# project class name → integer class ID (for YOLO output)
CLASS_ID = {
    "dent": 0, "scratch": 1, "crack": 2,
    "broken_lamp": 3, "shattered_glass": 4, "flat_tyre": 5,
}

PROJECT_CLASSES  = ["dent", "scratch", "crack", "broken_lamp", "shattered_glass", "flat_tyre"]
SEVERITY_BINS    = [0.0, 0.02, 0.08, 1.0]
SEVERITY_LABELS  = ["Minor", "Moderate", "Severe"]

PALETTE = {
    "dent":            "#2563EB",
    "scratch":         "#10B981",
    "crack":           "#F59E0B",
    "broken_lamp":     "#EF4444",
    "shattered_glass": "#8B5CF6",
    "flat_tyre":       "#EC4899",
    "exclude":         "#9CA3AF",
}

# Helpers 

def find_files(root: Path, exts: tuple) -> list:
    """Recursively collect all files with the given extensions."""
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_via_record(val):
    return isinstance(val, dict) and ("filename" in val or "name" in val) and "regions" in val


def _normalise_regions(via_regions: list) -> list:
    """Convert VIA region dicts to flat {all_x, all_y, vn_class} format."""
    out = []
    for r in via_regions:
        if "all_x" in r and "class" in r:          # already flat
            out.append(r)
            continue
        shape = r.get("shape_attributes", {})
        attrs = r.get("region_attributes", {})
        xs = shape.get("all_points_x") or shape.get("all_x", [])
        ys = shape.get("all_points_y") or shape.get("all_y", [])
        vn = (attrs.get("class") or attrs.get("damage") or "").strip().lower()
        if xs and ys:
            out.append({"all_x": xs, "all_y": ys, "class": vn})
    return out


def _polygon_to_bbox(all_x, all_y, img_w, img_h):
    if len(all_x) < 2:
        return None
    bw = max(all_x) - min(all_x)
    bh = max(all_y) - min(all_y)
    if bw <= 0 or bh <= 0:
        return None
    xc = (min(all_x) + max(all_x)) / 2.0 / img_w
    yc = (min(all_y) + max(all_y)) / 2.0 / img_h
    return max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)),            max(0.0, min(1.0, bw / img_w)), max(0.0, min(1.0, bh / img_h))


def load_via_json_files(json_paths: list) -> dict:
    """
    Load one or more VIA annotation JSON files.
    Returns { stem: [{"vn_class":..., "x_center":..., ...}] }
    """
    stem_to_records = {}
    for jp in json_paths:
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error("Cannot read %s: %s", jp, e)
            continue

        # Unwrap _via_img_metadata if present
        if "_via_img_metadata" in data:
            data = data["_via_img_metadata"]

        for key, val in data.items():
            if key.startswith("_via"):
                continue
            if not _is_via_record(val):
                continue
            stem    = Path(val.get("filename") or val.get("name", "")).stem
            regions = _normalise_regions(val.get("regions", []))
            stem_to_records[stem] = regions

    log.info("Loaded VIA annotations for %d images from %d file(s)",
             len(stem_to_records), len(json_paths))
    return stem_to_records


def via_regions_to_records(stem: str, regions: list, img_w: int, img_h: int) -> list:
    """Convert normalised region list → EDA record dicts (one per instance)."""
    records = []
    for r in regions:
        vn  = r.get("class", "").strip().lower()
        proj = VN_TO_PROJECT.get(vn)
        if proj is None or proj == "exclude":
            continue
        cid  = CLASS_ID.get(proj, -1)
        if cid == -1:
            continue
        bbox = _polygon_to_bbox(r["all_x"], r["all_y"], img_w, img_h)
        if bbox is None:
            continue
        xc, yc, bw, bh = bbox
        records.append({
            "vn_class":    vn,
            "class_id":    cid,
            "project_class": proj,
            "x_center":    xc,
            "y_center":    yc,
            "width":       bw,
            "height":      bh,
            "bbox_area":   bw * bh,
            "source_file": stem,
        })
    return records


def pil_size(path: Path):
    """Return (width, height) or None if corrupt."""
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


# Section 1: Load dataset

def load_dataset(data_dir: Path, annotation_files: list = None):
    log.info("=" * 60)
    log.info("STEP 1 — Loading dataset from %s", data_dir)
    log.info("=" * 60)

    img_paths = find_files(data_dir, (".jpg", ".jpeg", ".png"))
    img_stems = {p.stem: p for p in img_paths}
    log.info("Found %d image files", len(img_paths))

    # Load VIA JSON annotations
    if annotation_files:
        json_paths = [Path(p) for p in annotation_files]
    else:
        json_paths = find_files(data_dir, (".json",))
        if not json_paths:
            log.error("No annotation JSON files found. Use --annotation_files to specify them.")
            sys.exit(1)

    stem_to_regions = load_via_json_files(json_paths)
    ann_stems       = stem_to_regions   # dict of stem → regions (mirrors old ann_stems shape)

    # Orphan analysis
    img_set     = set(img_stems)
    ann_set     = set(stem_to_regions)
    orphan_imgs = img_set - ann_set
    orphan_anns = ann_set - img_set
    paired      = img_set & ann_set

    log.info("Paired (image + annotation): %d", len(paired))
    log.info("Orphan images (no annotation): %d", len(orphan_imgs))
    log.info("Orphan annotations (no image): %d", len(orphan_anns))

    # Build records — need image dimensions for normalised polygon → bbox conversion
    records = []
    for stem in sorted(paired):
        img_p = img_stems[stem]
        try:
            with Image.open(img_p) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 640, 640   # fallback if image unreadable

        records.extend(via_regions_to_records(stem, stem_to_regions[stem], img_w, img_h))

    df = pd.DataFrame(records)
    log.info("Total annotated instances: %d", len(df))

    return df, img_stems, ann_stems, orphan_imgs, orphan_anns


# Section 2: Class distribution

def analyse_classes(df: pd.DataFrame, output_dir: Path):
    log.info("=" * 60)
    log.info("STEP 2 — Class distribution analysis")
    log.info("=" * 60)

    # project_class already set during load; compute vn_class distribution too
    if "project_class" not in df.columns:
        df["project_class"] = df["class_id"].map(
            {v: k for k, v in CLASS_ID.items()})

    native_counts  = df["vn_class"].value_counts().sort_values(ascending=False)                      if "vn_class" in df.columns else df["class_id"].value_counts()
    project_counts = df["project_class"].value_counts()
    project_counts = project_counts.reindex(PROJECT_CLASSES, fill_value=0)

    excluded_count = 0   # exclusions already dropped during load

    log.info("\nVietnamese source class distribution:")
    for cls, cnt in native_counts.items():
        pct = 100 * cnt / len(df)
        log.info("  %-20s  %6d  (%.1f%%)", cls, cnt, pct)

    log.info("\nProject class distribution (after remapping):")
    for cls in PROJECT_CLASSES:
        cnt = project_counts[cls]
        pct = 100 * cnt / project_counts.sum()
        log.info("  %-20s  %6d  (%.1f%%)", cls, cnt, pct)
    log.info("  %-20s  %6d  (excluded)", "other/background", excluded_count)

    imbalance_ratio = project_counts.max() / project_counts.min()
    log.info("\nImbalance ratio (max/min): %.2f : 1", imbalance_ratio)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VehiDE — Class Distribution", fontsize=14, fontweight="bold")

    colors_project = [PALETTE[c] for c in PROJECT_CLASSES]
    bars = axes[0].bar(PROJECT_CLASSES, [project_counts[c] for c in PROJECT_CLASSES],
                       color=colors_project, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Instance Count per Project Class")
    axes[0].set_ylabel("Number of instances")
    axes[0].set_xticklabels(PROJECT_CLASSES, rotation=30, ha="right")
    for bar, cnt in zip(bars, [project_counts[c] for c in PROJECT_CLASSES]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(cnt), ha="center", va="bottom", fontsize=9)

    wedge_sizes  = [project_counts[c] for c in PROJECT_CLASSES]
    wedge_colors = colors_project
    axes[1].pie(wedge_sizes, labels=PROJECT_CLASSES, colors=wedge_colors,
                autopct="%1.1f%%", startangle=140,
                textprops={"fontsize": 9})
    axes[1].set_title("Class Proportion (%)")

    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved class_distribution.png")

    return df, project_counts


# Section 3: Bounding box area & severity proxy

def analyse_bbox(df: pd.DataFrame, output_dir: Path):
    log.info("=" * 60)
    log.info("STEP 3 — Bounding box area & severity proxy")
    log.info("=" * 60)

    df_proj = df.copy()
    df_proj["severity"] = pd.cut(
        df_proj["bbox_area"],
        bins=SEVERITY_BINS,
        labels=SEVERITY_LABELS,
        include_lowest=True,
    )

    severity_counts = df_proj["severity"].value_counts().reindex(SEVERITY_LABELS)
    log.info("\nSeverity proxy distribution:")
    for sev, cnt in severity_counts.items():
        pct = 100 * cnt / len(df_proj)
        log.info("  %-10s  %6d  (%.1f%%)", sev, cnt, pct)

    area_stats = df_proj["bbox_area"].describe()
    log.info("\nBounding box area statistics:")
    for stat, val in area_stats.items():
        log.info("  %-10s  %.5f", stat, val)

    # Per-class mean bbox area
    log.info("\nMean bbox area per class:")
    for cls in PROJECT_CLASSES:
        sub = df_proj[df_proj["project_class"] == cls]["bbox_area"]
        if len(sub) > 0:
            log.info("  %-20s  %.5f", cls, sub.mean())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("VehiDE — Bounding Box Area & Severity Proxy", fontsize=14, fontweight="bold")

    # Histogram of bbox areas
    axes[0].hist(df_proj["bbox_area"], bins=80, color="#2563EB", alpha=0.8, edgecolor="white")
    for boundary in SEVERITY_BINS[1:-1]:
        axes[0].axvline(boundary, color="red", linestyle="--", linewidth=1.5,
                        label=f"threshold {boundary}")
    axes[0].set_title("Bounding Box Area Distribution")
    axes[0].set_xlabel("Normalised bbox area (w x h)")
    axes[0].set_ylabel("Instance count")
    axes[0].legend(fontsize=8)

    # Severity proxy bar chart
    sev_colors = ["#10B981", "#F59E0B", "#EF4444"]
    bars = axes[1].bar(SEVERITY_LABELS, severity_counts.values, color=sev_colors, edgecolor="white")
    axes[1].set_title("Severity Proxy Distribution")
    axes[1].set_ylabel("Instance count")
    for bar, cnt in zip(bars, severity_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                     str(cnt), ha="center", va="bottom", fontsize=9)

    # Per-class mean bbox area
    class_means = [df_proj[df_proj["project_class"] == c]["bbox_area"].mean()
                   for c in PROJECT_CLASSES]
    clr = [PALETTE[c] for c in PROJECT_CLASSES]
    axes[2].barh(PROJECT_CLASSES, class_means, color=clr, edgecolor="white")
    axes[2].set_title("Mean Bbox Area per Class")
    axes[2].set_xlabel("Mean normalised area")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "bbox_area_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved bbox_area_distribution.png")

    # Aspect ratio distribution
    df_proj["aspect_ratio"] = df_proj["width"] / df_proj["height"].replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_proj["aspect_ratio"].dropna(), bins=60, color="#8B5CF6", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", label="Square (AR=1)")
    ax.set_title("Bounding Box Aspect Ratio Distribution")
    ax.set_xlabel("Width / Height")
    ax.set_ylabel("Instance count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "bbox_aspect_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved bbox_aspect_ratio.png")

    return df_proj, severity_counts


# Section 4: Instances per image

def analyse_instances_per_image(df: pd.DataFrame, output_dir: Path):
    log.info("=" * 60)
    log.info("STEP 4 — Instances per image")
    log.info("=" * 60)

    df_proj = df
    per_image = df_proj.groupby("source_file").size()

    log.info("Images with annotations: %d", len(per_image))
    log.info("Mean instances / image:  %.2f", per_image.mean())
    log.info("Median instances / image: %.1f", per_image.median())
    log.info("Max instances in 1 image: %d", per_image.max())

    counts_dist = per_image.value_counts().sort_index()
    log.info("\nDistribution:")
    for n, cnt in counts_dist.items():
        pct = 100 * cnt / len(per_image)
        log.info("  %2d instance(s): %5d images  (%.1f%%)", n, cnt, pct)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("VehiDE — Instances per Image", fontsize=14, fontweight="bold")

    axes[0].hist(per_image.values, bins=range(1, per_image.max() + 2),
                 color="#2563EB", alpha=0.85, edgecolor="white", align="left")
    axes[0].set_title("Distribution of Instance Counts")
    axes[0].set_xlabel("Instances per image")
    axes[0].set_ylabel("Number of images")
    axes[0].set_xticks(range(1, min(per_image.max() + 1, 15)))

    # Class co-occurrence (how often each class appears in the same image as another)
    df_multi = df_proj[df_proj["source_file"].isin(per_image[per_image > 1].index)]
    cooc = np.zeros((6, 6), dtype=int)
    for stem, group in df_multi.groupby("source_file"):
        classes = group["project_class"].tolist()
        for i, c1 in enumerate(PROJECT_CLASSES):
            for j, c2 in enumerate(PROJECT_CLASSES):
                if c1 in classes and c2 in classes:
                    cooc[i, j] += 1

    sns.heatmap(
        cooc, annot=True, fmt="d", cmap="Blues",
        xticklabels=PROJECT_CLASSES, yticklabels=PROJECT_CLASSES,
        ax=axes[1], linewidths=0.5,
    )
    axes[1].set_title("Class Co-occurrence Matrix\n(images with 2+ instances)")
    axes[1].set_xticklabels(PROJECT_CLASSES, rotation=35, ha="right", fontsize=8)
    axes[1].set_yticklabels(PROJECT_CLASSES, rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "instances_per_image.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved instances_per_image.png")

    # Save co-occurrence as separate heatmap
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cooc, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=PROJECT_CLASSES, yticklabels=PROJECT_CLASSES,
                ax=ax2, linewidths=0.5)
    ax2.set_title("Class Co-occurrence Heatmap")
    ax2.set_xticklabels(PROJECT_CLASSES, rotation=35, ha="right", fontsize=9)
    ax2.set_yticklabels(PROJECT_CLASSES, rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "class_cooccurrence_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved class_cooccurrence_heatmap.png")

    return per_image


# Section 5: Image resolution

def analyse_resolutions(img_stems: dict, output_dir: Path, sample_size=1000):
    log.info("=" * 60)
    log.info("STEP 5 — Image resolution analysis (sample: %d)", sample_size)
    log.info("=" * 60)

    stems = list(img_stems.keys())
    rng = np.random.default_rng(42)
    sampled = rng.choice(stems, size=min(sample_size, len(stems)), replace=False)

    widths, heights = [], []
    corrupt = []
    for stem in sampled:
        size = pil_size(img_stems[stem])
        if size is None:
            corrupt.append(stem)
        else:
            widths.append(size[0])
            heights.append(size[1])

    log.info("Corrupt images found in sample: %d", len(corrupt))
    log.info("Width  — mean: %.0f  median: %.0f  min: %d  max: %d",
             np.mean(widths), np.median(widths), min(widths), max(widths))
    log.info("Height — mean: %.0f  median: %.0f  min: %d  max: %d",
             np.mean(heights), np.median(heights), min(heights), max(heights))

    res_counter = Counter(zip(widths, heights))
    log.info("Top 5 resolutions:")
    for res, cnt in res_counter.most_common(5):
        log.info("  %dx%d: %d images", res[0], res[1], cnt)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("VehiDE — Image Resolution Analysis", fontsize=14, fontweight="bold")

    axes[0].hist(widths, bins=40, color="#2563EB", alpha=0.85, edgecolor="white")
    axes[0].axvline(640, color="red", linestyle="--", label="640px (YOLO target)")
    axes[0].set_title("Width Distribution")
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Image count")
    axes[0].legend(fontsize=8)

    axes[1].hist(heights, bins=40, color="#10B981", alpha=0.85, edgecolor="white")
    axes[1].axvline(640, color="red", linestyle="--", label="640px (YOLO target)")
    axes[1].set_title("Height Distribution")
    axes[1].set_xlabel("Pixels")
    axes[1].set_ylabel("Image count")
    axes[1].legend(fontsize=8)

    axes[2].scatter(widths, heights, alpha=0.3, s=8, color="#8B5CF6")
    axes[2].axvline(640, color="red", linestyle="--", linewidth=1)
    axes[2].axhline(640, color="red", linestyle="--", linewidth=1)
    axes[2].set_title("Width vs Height Scatter")
    axes[2].set_xlabel("Width (px)")
    axes[2].set_ylabel("Height (px)")

    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "image_resolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved image_resolution.png")

    return widths, heights, corrupt


# Section 6: Duplicate analysis

def analyse_duplicates(img_stems: dict, output_dir: Path):
    log.info("=" * 60)
    log.info("STEP 6 — Duplicate analysis (MD5 exact hash)")
    log.info("=" * 60)

    hash_map = defaultdict(list)
    for stem, path in img_stems.items():
        try:
            h = md5(path)
            hash_map[h].append(stem)
        except Exception as e:
            log.warning("Cannot hash %s: %s", stem, e)

    duplicates = {h: stems for h, stems in hash_map.items() if len(stems) > 1}
    dup_count  = sum(len(v) - 1 for v in duplicates.values())

    log.info("Unique MD5 hashes: %d", len(hash_map))
    log.info("Duplicate groups: %d", len(duplicates))
    log.info("Redundant images (copies to remove): %d", dup_count)

    report_lines = [f"Duplicate Groups Found: {len(duplicates)}\n"]
    for h, stems in list(duplicates.items())[:20]:
        report_lines.append(f"Hash {h[:12]}...: {stems}")
    if len(duplicates) > 20:
        report_lines.append(f"... and {len(duplicates) - 20} more groups")

    with open(output_dir / "duplicate_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    log.info("Saved duplicate_report.txt")

    return duplicates


# Section 7: Spatial analysis

def analyse_spatial(df: pd.DataFrame, output_dir: Path):
    log.info("=" * 60)
    log.info("STEP 7 — Spatial distribution of bounding box centres")
    log.info("=" * 60)

    df_proj = df

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("VehiDE — Spatial Distribution of Damage Centres per Class",
                 fontsize=13, fontweight="bold")

    for idx, cls in enumerate(PROJECT_CLASSES):
        ax  = axes[idx // 3][idx % 3]
        sub = df_proj[df_proj["project_class"] == cls]
        ax.scatter(sub["x_center"], sub["y_center"],
                   alpha=0.15, s=5, color=PALETTE[cls])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_title(f"{cls}\n(n={len(sub):,})", fontsize=10)
        ax.set_xlabel("x_center")
        ax.set_ylabel("y_center")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "spatial_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved spatial_distribution.png")


# Section 8: Summary JSON 

def save_summary(
    df, img_stems, ann_stems, orphan_imgs, orphan_anns,
    project_counts, severity_counts, per_image,
    widths, heights, duplicates, output_dir
):
    log.info("=" * 60)
    log.info("STEP 8 — Saving EDA summary")
    log.info("=" * 60)

    df_proj = df

    summary = {
        "dataset": "VehiDE",
        "total_images_found":       len(img_stems),
        "total_annotation_files":   len(ann_stems),
        "paired_images":            len(set(img_stems) & set(ann_stems)),
        "orphan_images":            len(orphan_imgs),
        "orphan_annotations":       len(orphan_anns),
        "total_instances_raw":      len(df),
        "total_instances_retained": len(df_proj),
        "excluded_instances":       int((df["project_class"] == "exclude").sum()),
        "class_distribution": {
            cls: int(project_counts.get(cls, 0)) for cls in PROJECT_CLASSES
        },
        "imbalance_ratio": round(
            float(project_counts.max() / project_counts.min()), 2
        ),
        "severity_proxy_distribution": {
            sev: int(cnt) for sev, cnt in severity_counts.items()
        },
        "bbox_area_stats": {
            stat: round(float(val), 6)
            for stat, val in df_proj["bbox_area"].describe().items()
        },
        "instances_per_image": {
            "mean":   round(float(per_image.mean()), 2),
            "median": float(per_image.median()),
            "max":    int(per_image.max()),
            "images_with_1_instance":  int((per_image == 1).sum()),
            "images_with_2_instances": int((per_image == 2).sum()),
            "images_with_3_instances": int((per_image == 3).sum()),
            "images_with_4plus":       int((per_image >= 4).sum()),
        },
        "image_resolution_sample": {
            "sample_size": len(widths),
            "width_mean":  round(float(np.mean(widths)), 1),
            "width_median": float(np.median(widths)),
            "width_min":   int(min(widths)),
            "width_max":   int(max(widths)),
            "height_mean": round(float(np.mean(heights)), 1),
            "height_median": float(np.median(heights)),
            "height_min":  int(min(heights)),
            "height_max":  int(max(heights)),
        },
        "duplicates": {
            "duplicate_groups": len(duplicates),
            "redundant_copies": sum(len(v) - 1 for v in duplicates.values()),
        },
    }

    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved eda_summary.json")

    # Save annotation DataFrame
    df.to_csv(output_dir / "annotation_dataframe.csv", index=False)
    log.info("Saved annotation_dataframe.csv  (%d rows)", len(df))

    # Save orphan lists
    with open(output_dir / "orphan_images.txt", "w") as f:
        f.write("\n".join(sorted(orphan_imgs)) + "\n")
    log.info("Saved orphan_images.txt  (%d entries)", len(orphan_imgs))

    # Print a readable summary to stdout
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"{k}: {v}")
    print("=" * 60)


# Main 

def main():
    parser = argparse.ArgumentParser(description="EDA for VehiDE dataset")
    parser.add_argument("--data_dir",         default="./data/vehide_raw",
                        help="Root directory containing raw VehiDE images")
    parser.add_argument("--output_dir",       default="./eda_outputs",
                        help="Directory for all EDA outputs")
    parser.add_argument("--annotation_files", nargs="+", default=None, metavar="JSON",
                        help="One or more VIA annotation JSON files. "
                             "Example: --annotation_files 0Train_via_annos.json 0Val_via_annos.json. "
                             "If omitted, all *.json files in data_dir are scanned automatically.")
    parser.add_argument("--res_sample",       default=1000, type=int,
                        help="Number of images to sample for resolution analysis")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        log.error("data_dir does not exist: %s", data_dir)
        sys.exit(1)

    # Resolve annotation file paths (relative to data_dir or cwd)
    annotation_files = None
    if args.annotation_files:
        annotation_files = []
        for p in args.annotation_files:
            resolved = Path(p)
            if not resolved.is_absolute() and (data_dir / resolved).exists():
                resolved = data_dir / resolved
            if not resolved.exists():
                log.error("Annotation file not found: %s", resolved)
                sys.exit(1)
            annotation_files.append(resolved)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 100

    df, img_stems, ann_stems, orphan_imgs, orphan_anns = load_dataset(data_dir, annotation_files)

    if df.empty:
        log.error("No annotation data loaded. Check --data_dir or --annotation_files.")
        sys.exit(1)

    df, project_counts              = analyse_classes(df, output_dir)
    df_proj, severity_counts        = analyse_bbox(df, output_dir)
    per_image                       = analyse_instances_per_image(df, output_dir)
    widths, heights, corrupt_sample = analyse_resolutions(img_stems, output_dir, args.res_sample)
    duplicates                      = analyse_duplicates(img_stems, output_dir)
    analyse_spatial(df, output_dir)

    save_summary(
        df, img_stems, ann_stems, orphan_imgs, orphan_anns,
        project_counts, severity_counts, per_image,
        widths, heights, duplicates, output_dir
    )

    log.info("EDA complete. All outputs saved to: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
