from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


SPLITS = ["train", "valid", "test"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a subset YOLOv8 dataset keeping only selected classes (and only images containing them)."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="data/roboflow/Player_detect.v1i.yolov8",
        help="Path to Roboflow YOLOv8 dataset root (contains train/valid/test and data.yaml).",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=["ball", "rim", "person"],
        help="Class names to keep (must exist in the dataset yaml names).",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="ball_rim_person",
        help="Suffix used for output split folders and yaml name.",
    )
    p.add_argument(
        "--keep-empty",
        action="store_true",
        help="If set, keep images even if they contain none of the selected classes (labels become empty). NOT recommended.",
    )
    return p.parse_args()


def find_split_folders(split_dir: Path) -> Tuple[Path, Path]:
    """
    Supports both:
      - split/images + split/labels
      - split_dir itself already contains images/labels
    """
    cand1_img = split_dir / "images"
    cand1_lbl = split_dir / "labels"
    if cand1_img.exists() and cand1_lbl.exists():
        return cand1_img, cand1_lbl

    # Some datasets use train_ball_rim/images already etc.
    # Fallback: search for immediate children named images/labels
    images = split_dir / "images"
    labels = split_dir / "labels"
    return images, labels


def load_dataset_yaml(dataset_root: Path) -> Tuple[Path, List[str]]:
    # Common names: data.yaml, dataset.yaml (Roboflow often uses data.yaml)
    for name in ["data.yaml", "dataset.yaml"]:
        y = dataset_root / name
        if y.exists():
            yaml_path = y
            break
    else:
        raise FileNotFoundError(f"No data.yaml found under: {dataset_root}")

    text = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # very small yaml parser just for 'names: [...]'
    names: List[str] = []
    for line in text:
        line_stripped = line.strip()
        if line_stripped.startswith("names:"):
            # expects: names: ['a','b',...]
            rhs = line_stripped.split("names:", 1)[1].strip()
            # remove brackets
            rhs = rhs.strip()
            if rhs.startswith("[") and rhs.endswith("]"):
                rhs = rhs[1:-1].strip()
            # split by comma
            # handle quotes
            parts = [p.strip() for p in rhs.split(",") if p.strip()]
            names = [p.strip().strip("'").strip('"') for p in parts]
            break

    if not names:
        raise RuntimeError(
            f"Could not parse class names from {yaml_path}. "
            f"Expected a line like: names: ['ball', 'rim', ...]"
        )

    return yaml_path, names


def read_yolo_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO labels: class cx cy w h (normalized).
    Returns list of tuples (cls, cx, cy, w, h).
    """
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            out.append((cls, cx, cy, w, h))
        except Exception:
            continue
    return out


def write_yolo_label_file(path: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (c, cx, cy, w, h) in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    classes_to_keep = [c.strip() for c in args.classes]
    suffix = args.suffix.strip()

    print(f"Dataset selected: {dataset_root}")

    yaml_path, names = load_dataset_yaml(dataset_root)
    name_to_id: Dict[str, int] = {n: i for i, n in enumerate(names)}

    missing = [c for c in classes_to_keep if c not in name_to_id]
    if missing:
        raise ValueError(
            f"These classes are not in dataset names: {missing}\n"
            f"Available: {names}"
        )

    kept_old_ids = [name_to_id[c] for c in classes_to_keep]
    old_to_new: Dict[int, int] = {old: new for new, old in enumerate(kept_old_ids)}

    # Output yaml path
    out_yaml = dataset_root / f"data_{suffix}.yaml"

    # Create each split
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"⚠️ Split folder not found: {split_dir} (skipping)")
            continue

        img_dir, lbl_dir = find_split_folders(split_dir)
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"⚠️ Missing images/labels in: {split_dir} (skipping)")
            continue

        out_split_dir = dataset_root / f"{split}_{suffix}"
        out_img_dir = out_split_dir / "images"
        out_lbl_dir = out_split_dir / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        # gather images
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

        kept = 0
        for img_path in images:
            label_path = lbl_dir / (img_path.stem + ".txt")
            rows = read_yolo_label_file(label_path)

            # filter to selected classes
            filtered = []
            for (cls, cx, cy, w, h) in rows:
                if cls in old_to_new:
                    filtered.append((old_to_new[cls], cx, cy, w, h))

            if (not filtered) and (not args.keep_empty):
                continue  # skip this image entirely

            # copy image
            shutil.copy2(img_path, out_img_dir / img_path.name)

            # write filtered label (possibly empty if keep_empty)
            write_yolo_label_file(out_lbl_dir / label_path.name, filtered)
            kept += 1

        print(f"✅ {split}: kept {kept} images -> {out_split_dir}")

    # Write output yaml (relative paths like Roboflow)
    # YOLO expects split paths that point to images folders.
    # Here we create split folders train_<suffix>/images etc.
    yaml_text = f"""train: ../train_{suffix}/images
val: ../valid_{suffix}/images
test: ../test_{suffix}/images

nc: {len(classes_to_keep)}
names: {classes_to_keep}
"""
    out_yaml.write_text(yaml_text, encoding="utf-8")
    print("\n✅ Done. You can now train with:")
    print(f"  data={out_yaml}")


if __name__ == "__main__":
    main()
