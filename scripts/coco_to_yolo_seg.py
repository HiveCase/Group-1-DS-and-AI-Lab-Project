import json, os
from pathlib import Path

REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # To be updated with correct mapping values

def convert(coco_json: str, out_dir: str):
    with open(coco_json) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    ann_map = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        ann_map.setdefault(img_id, []).append(ann)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for img_id, img_info in img_map.items():
        W, H = img_info["width"], img_info["height"]
        stem = Path(img_info["file_name"]).stem
        lines = []
        for ann in ann_map.get(img_id, []):
            cid = REMAP.get(ann["category_id"] - 1, -1)
            if cid == -1:
                continue
            seg = ann.get("segmentation", [])
            if not seg:
                x, y, w, h = ann["bbox"]
                pts = f"{(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}"
                lines.append(f"{cid} {pts}")
            else:
                coords = seg[0]
                norm = " ".join(
                    f"{coords[i]/W:.6f} {coords[i+1]/H:.6f}"
                    for i in range(0, len(coords), 2)
                )
                lines.append(f"{cid} {norm}")

        with open(f"{out_dir}/{stem}.txt", "w") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--coco_json", required=True)
    p.add_argument("--out_dir",   required=True)
    args = p.parse_args()
    convert(args.coco_json, args.out_dir)
    print("Done")
