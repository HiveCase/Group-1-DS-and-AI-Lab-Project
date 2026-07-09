import os
import sys
import json
import hashlib
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    print("imagehash not installed. Near-duplicate removal skipped. "
          "Install with: pip install imagehash")

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

#  Vietnamese → English class name mapping 
VN_TO_EN = {
    "mat_bo_phan": "lost_parts",
    "rach":        "torn_body",
    "mop_lom":     "dents",
    "tray_son":    "paint_scratches",
    "thung":       "puncture",
    "vo_kinh":     "broken_glass",
    "be_den":      "broken_lamp",
}

#  English intermediate → project class ID 
# Project classes: dent(0) scratch(1) crack(2) broken_lamp(3) shattered_glass(4) flat_tyre(5)
# Note: "crack" has no direct VN equivalent; "rach" (torn_body) and "tray_son"
# (paint_scratches) both map to scratch. "lost_parts" is excluded.
EN_TO_CLASS_ID = {
    "lost_parts":     -1,   # exclude — missing parts, not a visual damage class
    "torn_body":       1,   # → scratch (closest project class: surface tear/scratch)
    "dents":           0,   # → dent
    "paint_scratches": 1,   # → scratch
    "puncture":        5,   # → flat_tyre
    "broken_glass":    4,   # → shattered_glass
    "broken_lamp":     3,   # → broken_lamp
}

# Project class names indexed by ID (crack=2 has no VN source in this dataset)
CLASS_NAMES = ["dent", "scratch", "crack", "broken_lamp", "shattered_glass", "flat_tyre"]

# Processing log accumulator
LOG = {
    "corrupt_removed":            [],
    "orphan_images_removed":      [],
    "orphan_anns_removed":        [],
    "exact_dup_removed":          [],
    "near_dup_removed":           [],
    "pii_blurred":                defaultdict(list),
    "malformed_annotations":      [],
    "excluded_instances":         0,
    "total_instances_raw":        0,
    "total_instances_after_map":  0,
    "vn_class_counts":            defaultdict(int),
    "project_class_counts":       defaultdict(int),
    "split":                      {},
}

# Helpers 

def find_files(root: Path, exts: tuple) -> list:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def letterbox(img: Image.Image, size: int = 640) -> Image.Image:
    """Resize to (size x size) with grey letterbox padding."""
    img_rgb = img.convert("RGB")
    img_rgb.thumbnail((size, size), Image.LANCZOS)
    dw = size - img_rgb.size[0]
    dh = size - img_rgb.size[1]
    padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
    return ImageOps.expand(img_rgb, padding, fill=(114, 114, 114))


# JSON Annotation loading
#
# Three annotation formats are supported:
#
# Format 1 — Per-image JSON (one file per image):
#   { "name": "<filename>.jpg", "regions": [{"all_x":[...], "all_y":[...], "class":"rach"}, ...] }
#
# Format 2 — Simple combined JSON (filename → regions):
#   { "<filename>.jpg": { "regions": [{"all_x":[...], "all_y":[...], "class":"rach"}] }, ... }
#
# Format 3 — VIA (VGG Image Annotator) export — the format used by VehiDE:
#   {
#     "_via_settings": {...},                   ← optional VIA metadata, ignored
#     "_via_img_metadata": {                    ← may or may not be present
#       "<filename><size>": {
#         "filename": "<filename>.jpg",
#         "size": <bytes>,
#         "regions": [
#           {
#             "shape_attributes": {
#               "name": "polygon",
#               "all_points_x": [...],
#               "all_points_y": [...]
#             },
#             "region_attributes": { "class": "<vn_class>" }
#           }
#         ],
#         "file_attributes": {}
#       }
#     }
#   }
#
#   VIA files can also be exported WITHOUT the _via_img_metadata wrapper — just the
#   inner dict directly at the top level. Both variants are detected automatically.

def _is_via_record(val: dict) -> bool:
    """Return True if val looks like a VIA image record (has 'filename' and 'regions')."""
    return isinstance(val, dict) and "filename" in val and "regions" in val


def _normalise_via_regions(via_regions: list) -> list:
    """
    Convert VIA region dicts to the project's internal format:
        { "all_x": [...], "all_y": [...], "class": "<vn_class>" }

    Handles:
      - VIA shape_attributes + region_attributes  (VIA export)
      - Already-normalised format with all_x / all_y / class (per-image JSON or simple combined)
    """
    normalised = []
    for r in via_regions:
        #  Already in internal format 
        if "all_x" in r and "all_y" in r and "class" in r:
            normalised.append(r)
            continue

        # VIA export format
        shape = r.get("shape_attributes", {})
        attrs = r.get("region_attributes", {})

        xs = shape.get("all_points_x") or shape.get("all_x", [])
        ys = shape.get("all_points_y") or shape.get("all_y", [])

        # class field: try common VIA attribute names
        vn_class = (attrs.get("class") or attrs.get("damage") or
                    attrs.get("label") or attrs.get("type") or "")
        vn_class = str(vn_class).strip().lower()

        if xs and ys:
            normalised.append({"all_x": xs, "all_y": ys, "class": vn_class})
        # else: empty polygon — will be caught by parse_regions later

    return normalised


def _parse_single_json(data: dict, source_name: str) -> dict:
    """
    Parse a loaded JSON object into { stem: [normalised_region, ...] }.
    Handles all three formats described above.
    """
    stem_to_regions = {}

    # Format 1: single-image per-file
    if "name" in data and "regions" in data:
        stem = Path(data["name"]).stem
        stem_to_regions[stem] = _normalise_via_regions(data["regions"])
        return stem_to_regions

    # VIA with _via_img_metadata wrapper
    if "_via_img_metadata" in data:
        inner = data["_via_img_metadata"]
        log.info("  Detected VIA format with _via_img_metadata wrapper in %s", source_name)
        for key, val in inner.items():
            if _is_via_record(val):
                stem = Path(val["filename"]).stem
                stem_to_regions[stem] = _normalise_via_regions(val.get("regions", []))
        return stem_to_regions

    # ── VIA without wrapper OR simple combined format ───────────────────
    # Keys are either "<filename><size>" (VIA) or "<filename>.jpg" (simple combined).
    # Distinguish by checking whether values have a "filename" field (VIA) or a
    # "regions" field directly (simple combined).
    for key, val in data.items():
        if key.startswith("_via"):          # skip VIA metadata keys
            continue
        if not isinstance(val, dict):
            continue

        if _is_via_record(val):             # VIA without wrapper
            stem = Path(val["filename"]).stem
            stem_to_regions[stem] = _normalise_via_regions(val.get("regions", []))

        elif "regions" in val:              # simple combined format
            stem = Path(key).stem
            stem_to_regions[stem] = _normalise_via_regions(val["regions"])

    if stem_to_regions:
        log.info("  Detected VIA/combined format (no wrapper) in %s", source_name)
    return stem_to_regions


def load_annotation_files(json_paths: list) -> dict:
    """
    Load one or more annotation JSON files (any supported format) and merge
    all image → regions mappings into a single dict.

    json_paths: list of Path objects
    Returns: { stem: [normalised_region, ...], ... }
    """
    merged = {}
    for jp in json_paths:
        log.info("Loading annotation file: %s", jp.name)
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error("Cannot read %s: %s", jp.name, e)
            continue

        parsed = _parse_single_json(data, jp.name)
        overlap = set(merged) & set(parsed)
        if overlap:
            log.warning("  %d stems overlap with previous files — keeping latest", len(overlap))
        merged.update(parsed)
        log.info("  +%d images  (running total: %d)", len(parsed), len(merged))

    return merged


def load_all_annotations(data_dir: Path, annotation_files: list = None) -> dict:
    """
    Build a mapping: stem → list of normalised region dicts.

    If annotation_files is provided (list of Paths), load from those files.
    Otherwise scan data_dir for *.json files automatically.

    Returns:
        { "image_stem": [{"all_x": [...], "all_y": [...], "class": "..."}, ...], ... }
    """
    if annotation_files:
        return load_annotation_files(annotation_files)

    # Auto-scan data_dir for JSON files
    json_paths = find_files(data_dir, (".json",))
    log.info("Auto-scan found %d JSON file(s) in %s", len(json_paths), data_dir)
    if not json_paths:
        log.error("No JSON annotation files found. Use --annotation_files to specify them.")
        sys.exit(1)
    return load_annotation_files(json_paths)


# Polygon → YOLO bounding box 

def polygon_to_yolo_bbox(all_x: list, all_y: list, img_w: int, img_h: int) -> tuple:
    """
    Convert a polygon defined by all_x / all_y point lists to a
    YOLO-format normalised bounding box.

    Returns: (x_center, y_center, width, height) all in [0, 1].
    Returns None if the polygon has fewer than 2 points or zero area.
    """
    if len(all_x) < 2 or len(all_y) < 2:
        return None

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        return None

    x_center = (x_min + x_max) / 2.0 / img_w
    y_center  = (y_min + y_max) / 2.0 / img_h
    width     = bw / img_w
    height    = bh / img_h

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center  = max(0.0, min(1.0, y_center))
    width     = max(0.0, min(1.0, width))
    height    = max(0.0, min(1.0, height))

    if width == 0 or height == 0:
        return None

    return (x_center, y_center, width, height)


# Per-image annotation parsing 

def parse_regions(regions: list, img_w: int, img_h: int, stem: str) -> tuple:
    """
    Parse a list of region dicts for one image.

    Returns:
        valid_instances: list of (class_id, x_center, y_center, width, height)
        excluded:        count of instances excluded (unknown class or zero-area bbox)
        malformed:       list of warning strings
    """
    valid     = []
    excluded  = 0
    malformed = []

    for i, region in enumerate(regions):
        all_x = region.get("all_x", [])
        all_y = region.get("all_y", [])
        vn_class = region.get("class", "").strip().lower()

        #  Vietnamese → English
        en_class = VN_TO_EN.get(vn_class)
        if en_class is None:
            malformed.append(f"{stem} region {i}: unknown VN class '{vn_class}'")
            excluded += 1
            LOG["vn_class_counts"]["UNKNOWN"] += 1
            continue

        LOG["vn_class_counts"][vn_class] += 1

        #  English → project class ID
        class_id = EN_TO_CLASS_ID.get(en_class, -1)
        if class_id == -1:
            excluded += 1
            continue

        #  Polygon → YOLO bbox
        if len(all_x) != len(all_y):
            malformed.append(f"{stem} region {i}: all_x and all_y length mismatch "
                             f"({len(all_x)} vs {len(all_y)})")
            excluded += 1
            continue

        bbox = polygon_to_yolo_bbox(all_x, all_y, img_w, img_h)
        if bbox is None:
            malformed.append(f"{stem} region {i}: zero-area polygon for class '{vn_class}'")
            excluded += 1
            continue

        valid.append((class_id, *bbox))
        LOG["project_class_counts"][CLASS_NAMES[class_id]] += 1

    return valid, excluded, malformed


def write_yolo_annotation(instances: list, out_path: Path):
    """Write YOLO .txt annotation file."""
    with open(out_path, "w", encoding="utf-8") as f:
        for (cid, xc, yc, w, h) in instances:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# PII Detection

FACE_CASCADE_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PROFILE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"

def _load_cascade(path):
    if os.path.exists(path):
        return cv2.CascadeClassifier(path)
    return None

FACE_CASCADE    = _load_cascade(FACE_CASCADE_PATH)
PROFILE_CASCADE = _load_cascade(PROFILE_CASCADE_PATH)


def is_plate_candidate(bbox, img_w, img_h):
    x, y, w, h = bbox
    if h == 0:
        return False
    return 2.5 <= (w / h) <= 6.0 and 0.2 <= (w * h / (img_w * img_h) * 100) <= 5.0


def detect_and_blur_pii(img_cv2: np.ndarray) -> tuple:
    h, w = img_cv2.shape[:2]
    gray  = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    out   = img_cv2.copy()
    k     = (51, 51)
    pii   = {"faces": 0, "license_plates": 0}

    if FACE_CASCADE is not None:
        for (fx, fy, fw, fh) in FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
            out[fy:fy+fh, fx:fx+fw] = cv2.GaussianBlur(out[fy:fy+fh, fx:fx+fw], k, 0)
            pii["faces"] += 1

    if PROFILE_CASCADE is not None:
        for (px, py, pw, ph) in PROFILE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
            out[py:py+ph, px:px+pw] = cv2.GaussianBlur(out[py:py+ph, px:px+pw], k, 0)
            pii["faces"] += 1

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        if is_plate_candidate((x, y, rw, rh), w, h):
            out[y:y+rh, x:x+rw] = cv2.GaussianBlur(out[y:y+rh, x:x+rw], k, 0)
            pii["license_plates"] += 1

    return out, pii


# Pipeline steps

def step_integrity_check(img_paths: list) -> list:
    log.info("STEP 1 — Integrity check (%d images)", len(img_paths))
    valid = []
    for p in tqdm(img_paths, desc="Checking integrity", unit="img"):
        if is_valid_image(p):
            valid.append(p)
        else:
            LOG["corrupt_removed"].append(str(p))
    log.info("  Corrupt removed: %d  |  Remaining: %d",
             len(img_paths) - len(valid), len(valid))
    return valid


def step_orphan_removal(img_paths: list, stem_to_regions: dict) -> tuple:
    log.info("STEP 2 — Orphan removal")
    img_stems = {p.stem: p for p in img_paths}

    ann_stems_set = set(stem_to_regions.keys())
    img_stems_set = set(img_stems.keys())

    paired       = img_stems_set & ann_stems_set
    orphan_img   = img_stems_set - ann_stems_set
    orphan_ann   = ann_stems_set - img_stems_set

    LOG["orphan_images_removed"].extend(sorted(orphan_img))
    LOG["orphan_anns_removed"].extend(sorted(orphan_ann))

    log.info("  Paired: %d  |  Orphan images: %d  |  Orphan annotations: %d",
             len(paired), len(orphan_img), len(orphan_ann))

    valid_imgs = [img_stems[s] for s in sorted(paired)]
    return valid_imgs


def step_exact_dedup(img_paths: list) -> list:
    log.info("STEP 3 — Exact duplicate removal (MD5)")
    seen   = {}
    unique = []
    for p in tqdm(img_paths, desc="MD5 hashing", unit="img"):
        h = md5(p)
        if h in seen:
            LOG["exact_dup_removed"].append(str(p))
        else:
            seen[h] = p
            unique.append(p)
    log.info("  Duplicates removed: %d  |  Remaining: %d",
             len(img_paths) - len(unique), len(unique))
    return unique


def step_near_dedup(img_paths: list, threshold: int = 8) -> list:
    if not PHASH_AVAILABLE:
        log.info("STEP 4 — Near-duplicate removal SKIPPED (imagehash not installed)")
        return img_paths

    log.info("STEP 4 — Near-duplicate removal (pHash, threshold=%d bits)", threshold)
    hashes  = {}
    unique  = []
    removed = 0
    for p in tqdm(img_paths, desc="pHashing", unit="img"):
        try:
            with Image.open(p) as img:
                ph = imagehash.phash(img)
        except Exception:
            unique.append(p)
            continue
        is_dup = any(ph - ref <= threshold for ref in hashes)
        if is_dup:
            LOG["near_dup_removed"].append(str(p))
            removed += 1
        else:
            hashes[ph] = p
            unique.append(p)
    log.info("  Near-duplicates removed: %d  |  Remaining: %d", removed, len(unique))
    return unique


def step_preprocess_single(
    img_path: Path,
    regions: list,
    out_img_path: Path,
    out_ann_path: Path,
    img_size: int,
    blur_pii: bool,
) -> dict:
    """
    Process one image-annotation pair end-to-end.
    Returns a stats dict.
    """
    stats = {
        "stem":           img_path.stem,
        "excluded":       0,
        "retained":       0,
        "faces_blurred":  0,
        "plates_blurred": 0,
        "skipped":        False,
        "skip_reason":    "",
    }

    # Load image to get dimensions
    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        stats["skipped"]     = True
        stats["skip_reason"] = str(e)
        return stats

    img_w, img_h = img_pil.size
    LOG["total_instances_raw"] += len(regions)

    # Parse regions → YOLO instances
    instances, excluded, malformed = parse_regions(regions, img_w, img_h, img_path.stem)
    LOG["malformed_annotations"].extend(malformed)
    LOG["excluded_instances"] += excluded
    LOG["total_instances_after_map"] += len(instances)
    stats["excluded"] = excluded
    stats["retained"] = len(instances)

    # PII blurring
    if blur_pii:
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_cv, pii = detect_and_blur_pii(img_cv)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        stats["faces_blurred"]  = pii["faces"]
        stats["plates_blurred"] = pii["license_plates"]
        if pii["faces"] > 0:
            LOG["pii_blurred"]["faces"].append(img_path.stem)
        if pii["license_plates"] > 0:
            LOG["pii_blurred"]["license_plates"].append(img_path.stem)

    # Letterbox resize
    img_resized = letterbox(img_pil, size=img_size)

    # Write outputs
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    img_resized.save(out_img_path, "JPEG", quality=95)
    write_yolo_annotation(instances, out_ann_path)

    return stats


def step_split(processed_stems: list, dominant_map: dict, seed: int) -> tuple:
    log.info("STEP — Stratified train/val/test split (70/15/15, seed=%d)", seed)

    stems  = [s for s in processed_stems if s in dominant_map]
    labels = [dominant_map[s] for s in stems]
    n      = len(stems)

    # Small-dataset fallback
    # sklearn requires at least 2 samples per class for stratified split, and
    # at least 1 sample in each resulting split. For small datasets (typically
    # during development or partial runs) we fall back to simple sequential
    # assignment rather than crashing.
    #
    # Thresholds:
    #   < 3  images → all go to train; val and test are empty
    #   3-9  images → simple (non-stratified) random split
    #  >= 10 images → full stratified split (normal production path)

    if n < 3:
        log.warning(
            "Only %d image(s) available. Assigning all to train split. "
            "Val and test will be empty. Run on the full dataset for a proper split.", n
        )
        return stems, [], []

    if n < 10:
        log.warning(
            "%d images is too few for a stratified split. "
            "Falling back to simple random split (no stratification).", n
        )
        rng          = np.random.default_rng(seed)
        shuffled     = list(rng.permutation(stems))
        n_train      = max(1, int(n * 0.70))
        n_val        = max(1, int(n * 0.15)) if n - n_train >= 2 else 0
        train        = shuffled[:n_train]
        val          = shuffled[n_train: n_train + n_val]
        test         = shuffled[n_train + n_val:]
        log.info("  Train: %d  |  Val: %d  |  Test: %d", len(train), len(val), len(test))
        return train, val, test

    # Normal stratified split
    # Check whether stratification is feasible (every class needs >= 2 samples).
    from collections import Counter
    label_counts = Counter(labels)
    can_stratify = all(c >= 2 for c in label_counts.values())

    if not can_stratify:
        log.warning(
            "Some classes have fewer than 2 samples; falling back to "
            "non-stratified split to avoid sklearn error."
        )
        stratify_arg = None
    else:
        stratify_arg = labels

    train, temp, y_train, y_temp = train_test_split(
        stems, labels, test_size=0.30, random_state=seed, stratify=stratify_arg
    )

    # Second split: val and test from temp.
    # Need at least 2 samples in temp for a 50/50 split.
    if len(temp) < 2:
        log.warning(
            "temp split has only %d sample(s) after first split; "
            "assigning all of temp to val, test will be empty.", len(temp)
        )
        val, test = temp, []
    else:
        can_stratify_temp = (stratify_arg is not None and
                             all(c >= 2 for c in Counter(y_temp).values()))
        val, test, _, _ = train_test_split(
            temp, y_temp, test_size=0.50, random_state=seed,
            stratify=y_temp if can_stratify_temp else None
        )

    log.info("  Train: %d  |  Val: %d  |  Test: %d", len(train), len(val), len(test))

    # Leakage check
    for a_name, a_set, b_name, b_set in [
        ("train", set(train), "val",  set(val)),
        ("train", set(train), "test", set(test)),
        ("val",   set(val),   "test", set(test)),
    ]:
        leak = a_set & b_set
        if leak:
            log.error("LEAKAGE: %s / %s overlap = %d items", a_name, b_name, len(leak))
            sys.exit(1)
    log.info("  Leakage check PASSED")
    return train, val, test


def step_verify_hash_leakage(train, val, test, proc_dir: Path):
    log.info("STEP — Cross-split hash leakage verification")

    def get_hashes(stems, split):
        return {s: md5(proc_dir / "images" / split / f"{s}.jpg")
                for s in stems
                if (proc_dir / "images" / split / f"{s}.jpg").exists()}

    train_h = set(get_hashes(train, "train").values())
    val_h   = set(get_hashes(val,   "val").values())
    test_h  = set(get_hashes(test,  "test").values())

    if (train_h & val_h) or (train_h & test_h) or (val_h & test_h):
        log.error("Hash leakage detected across splits!")
    else:
        log.info("  All clear — zero hash matches across splits")


def write_damage_yaml(output_dir: Path):
    content = f"""# YOLO dataset configuration — VehiDE Vehicle Damage Detection
# Generated by preprocess_images.py
# Vietnamese source classes mapped to project taxonomy

path: {output_dir.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

# Source class mapping (Vietnamese → English → project class)
# mat_bo_phan → lost_parts     → EXCLUDED
# rach        → torn_body      → scratch  (id 1)
# mop_lom     → dents          → dent     (id 0)
# tray_son    → paint_scratch  → scratch  (id 1)
# thung       → puncture       → flat_tyre(id 5)
# vo_kinh     → broken_glass   → shattered_glass (id 4)
# be_den      → broken_lamp    → broken_lamp     (id 3)
# Note: crack (id 2) has no direct Vietnamese source in this dataset
"""
    with open(output_dir / "damage.yaml", "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Saved damage.yaml")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="VehiDE image preprocessing pipeline (JSON polygon annotations)"
    )
    parser.add_argument("--data_dir",         required=True,
                        help="Root directory containing images")
    parser.add_argument("--output_dir",       required=True,
                        help="Output directory for processed dataset")
    parser.add_argument("--annotation_files", nargs="+", default=None, metavar="JSON",
                        help="One or more annotation JSON files. Supports per-image JSON, "
                             "simple combined JSON, and VIA (VGG Image Annotator) export. "
                             "Example: --annotation_files 0Train_via_annos.json 0Val_via_annos.json. "
                             "If omitted, all *.json files in data_dir are scanned automatically.")
    parser.add_argument("--img_size",         default=640,   type=int)
    parser.add_argument("--seed",             default=42,    type=int)
    parser.add_argument("--blur_pii",         action="store_true",
                        help="Enable PII detection and blurring")
    parser.add_argument("--phash_thresh",     default=8,     type=int,
                        help="pHash Hamming threshold for near-dedup")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve annotation file paths
    annotation_files = None
    if args.annotation_files:
        annotation_files = []
        for p in args.annotation_files:
            resolved = Path(p)
            if not resolved.is_absolute():
                if (data_dir / resolved).exists():
                    resolved = data_dir / resolved
            if not resolved.exists():
                log.error("Annotation file not found: %s", resolved)
                sys.exit(1)
            annotation_files.append(resolved)

    # Load all annotations first 
    log.info("=" * 60)
    log.info("LOADING ANNOTATIONS")
    log.info("=" * 60)
    stem_to_regions = load_all_annotations(data_dir, annotation_files)

    #  Collect image files 
    img_paths = find_files(data_dir, (".jpg", ".jpeg", ".png"))
    log.info("Raw images found: %d", len(img_paths))

    # Pipeline
    img_paths = step_integrity_check(img_paths)
    img_paths = step_orphan_removal(img_paths, stem_to_regions)
    img_paths = step_exact_dedup(img_paths)
    img_paths = step_near_dedup(img_paths, threshold=args.phash_thresh)

    log.info("After all filters: %d images remain", len(img_paths))

    # Process each image
    temp_dir     = output_dir / "_temp"
    dominant_map = {}   # stem → most frequent project class_id (for stratification)
    all_stats    = []

    log.info("=" * 60)
    log.info("PREPROCESSING %d image-annotation pairs", len(img_paths))
    log.info("=" * 60)

    for img_p in tqdm(img_paths, desc="Preprocessing", unit="img"):
        regions   = stem_to_regions.get(img_p.stem, [])
        temp_img  = temp_dir / "images" / f"{img_p.stem}.jpg"
        temp_ann  = temp_dir / "labels" / f"{img_p.stem}.txt"

        stat = step_preprocess_single(
            img_p, regions, temp_img, temp_ann,
            args.img_size, args.blur_pii
        )
        all_stats.append(stat)

        # Build dominant class for stratification
        # Re-parse quickly to get class ids (already validated above)
        try:
            w, h = Image.open(img_p).size
        except Exception:
            w, h = 640, 640
        instances, _, _ = parse_regions(regions, w, h, img_p.stem)
        if instances:
            counts = defaultdict(int)
            for (cid, *_) in instances:
                counts[cid] += 1
            dominant_map[img_p.stem] = max(counts, key=counts.get)
        else:
            dominant_map[img_p.stem] = 0   # background-only → default to dent bucket

    # Split
    processed_stems = [img_p.stem for img_p in img_paths]
    train, val, test = step_split(processed_stems, dominant_map, args.seed)

    # Organise into final split folders
    log.info("Organising into split folders...")
    for split_name, stems in [("train", train), ("val", val), ("test", test)]:
        for stem in tqdm(stems, desc=f"Copying {split_name}", unit="img"):
            src_img = temp_dir / "images" / f"{stem}.jpg"
            src_ann = temp_dir / "labels" / f"{stem}.txt"
            dst_img = output_dir / "images" / split_name / f"{stem}.jpg"
            dst_ann = output_dir / "labels" / split_name / f"{stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_ann.parent.mkdir(parents=True, exist_ok=True)
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_ann.exists():
                shutil.copy2(src_ann, dst_ann)

    shutil.rmtree(temp_dir, ignore_errors=True)

    # Leakage hash verification
    step_verify_hash_leakage(train, val, test, output_dir)

    # Split file lists
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for split_name, stems in [("train", train), ("val", val), ("test", test)]:
        with open(splits_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            for s in stems:
                f.write(f"./images/{split_name}/{s}.jpg\n")

    # YAML and checksums 
    write_damage_yaml(output_dir)

    cksum_lines = []
    for split_name in ["train", "val", "test"]:
        imgs = sorted((output_dir / "images" / split_name).glob("*.jpg"))[:10]
        for p in imgs:
            cksum_lines.append(f"{md5(p)}  {p.name}  [{split_name}]")
    with open(output_dir / "checksums.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cksum_lines))

    # Class stats
    class_stats = {
        "vn_source_counts":     dict(LOG["vn_class_counts"]),
        "project_class_counts": dict(LOG["project_class_counts"]),
        "vn_to_en_mapping":     VN_TO_EN,
        "en_to_class_id":       EN_TO_CLASS_ID,
        "project_class_names":  CLASS_NAMES,
        "note_crack_class":     ("crack (class id 2) has no direct Vietnamese source "
                                 "in VehiDE. It will be populated from CarDD supplement."),
    }
    with open(output_dir / "class_stats.json", "w", encoding="utf-8") as f:
        json.dump(class_stats, f, indent=2)

    # Processing log 
    LOG["split"] = {"train": len(train), "val": len(val), "test": len(test)}
    LOG["pii_blurred"] = {
        "faces":          len(LOG["pii_blurred"]["faces"]),
        "license_plates": len(LOG["pii_blurred"]["license_plates"]),
    }
    LOG["malformed_annotations"] = len(LOG["malformed_annotations"])

    with open(output_dir / "processing_log.json", "w", encoding="utf-8") as f:
        json.dump(dict(LOG), f, indent=2, default=str)

    # summary 
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output directory:             {output_dir.resolve()}")
    print(f"Corrupt images removed:       {len(LOG['corrupt_removed'])}")
    print(f"Orphan images removed:        {len(LOG['orphan_images_removed'])}")
    print(f"Exact duplicates removed:     {len(LOG['exact_dup_removed'])}")
    print(f"Near-duplicates removed:      {len(LOG['near_dup_removed'])}")
    print(f"Malformed annotation regions: {LOG['malformed_annotations']}")
    print(f"Total instances (raw):        {LOG['total_instances_raw']}")
    print(f"Excluded instances:           {LOG['excluded_instances']}")
    print(f"Retained instances:           {LOG['total_instances_after_map']}")
    print(f"Faces blurred:                {LOG['pii_blurred']['faces']}")
    print(f"License plates blurred:       {LOG['pii_blurred']['license_plates']}")
    print(f"\nVietnamese class counts (raw):")
    for vn, cnt in sorted(LOG["vn_class_counts"].items(), key=lambda x: -x[1]):
        en  = VN_TO_EN.get(vn, "unknown")
        cid = EN_TO_CLASS_ID.get(en, -1)
        proj = CLASS_NAMES[cid] if cid >= 0 else "EXCLUDED"
        print(f"  {vn:<20} ({en:<18}) → {proj:<18}  {cnt:>6} instances")
    print(f"\nProject class counts (after mapping):")
    for cls, cnt in sorted(LOG["project_class_counts"].items(), key=lambda x: -x[1]):
        print(f"  {cls:<20}  {cnt:>6}")
    print(f"\nSplit sizes:")
    print(f"  Train: {LOG['split']['train']} images")
    print(f"  Val:   {LOG['split']['val']} images")
    print(f"  Test:  {LOG['split']['test']} images")
    print("=" * 60)
    print(f"\nStart training with:")
    print(f"  yolo train data={output_dir}/damage.yaml model=yolo11m-seg.pt epochs=50")


if __name__ == "__main__":
    main()
