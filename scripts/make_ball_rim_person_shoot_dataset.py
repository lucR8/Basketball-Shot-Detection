from __future__ import annotations
from pathlib import Path
import shutil

# Cherche automatiquement le dataset dans data/roboflow/
ROBOFLOW_ROOT = Path("data/roboflow")

# Mapping old -> new (d'après votre data.yaml)
# names: ['Basketball-courts', 'ball', 'made', 'person', 'rim', 'shoot']
# ids:     0                 1       2       3        4      5
KEEP = {1: 0, 4: 1, 3: 2, 5: 3}  # ball->0, rim->1, person->2, shoot->3

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_dataset_dir() -> Path:
    if not ROBOFLOW_ROOT.exists():
        raise FileNotFoundError(f"Missing folder: {ROBOFLOW_ROOT.resolve()}")

    # On prend le premier dossier qui contient un data.yaml et un dossier train/
    candidates = []
    for d in ROBOFLOW_ROOT.iterdir():
        if d.is_dir() and (d / "data.yaml").exists():
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            f"No dataset folder with data.yaml found under {ROBOFLOW_ROOT.resolve()}"
        )

    # Si vous en avez plusieurs, prenez celui qui contient train/
    for d in candidates:
        if (d / "train").exists():
            return d

    # Sinon le 1er
    return candidates[0]


def label_has_keep_class(label_path: Path) -> bool:
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return False
    for line in txt.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        if cls in KEEP:
            return True
    return False


def remap_and_write_label(src_label: Path, dst_label: Path) -> None:
    txt = src_label.read_text(encoding="utf-8", errors="ignore").strip()
    out_lines = []
    if txt:
        for line in txt.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                old_cls = int(float(parts[0]))
            except Exception:
                continue
            if old_cls in KEEP:
                parts[0] = str(KEEP[old_cls])
                out_lines.append(" ".join(parts))

    dst_label.write_text(
        "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
    )


def process_split(dataset_dir: Path, split: str) -> None:
    split_dir = dataset_dir / split
    if not split_dir.exists():
        print(f"⚠️ Split dir not found: {split_dir} (skipping)")
        return

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(
            f"⚠️ Missing images/labels for '{split}': {images_dir} / {labels_dir} (skipping)"
        )
        return

    out_split = dataset_dir / f"{split}_ball_rim_person_shoot"
    out_images = out_split / "images"
    out_labels = out_split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    kept = 0
    # On parcourt les labels: c'est la source de vérité
    for lab in labels_dir.glob("*.txt"):
        if not label_has_keep_class(lab):
            continue

        stem = lab.stem

        # trouver l'image associée (peut être jpg/png/...)
        img_path = None
        for ext in IMG_EXTS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # fallback glob
            matches = list(images_dir.glob(stem + ".*"))
            matches = [m for m in matches if m.is_file() and is_image(m)]
            if matches:
                img_path = matches[0]

        if img_path is None:
            print(f"⚠️ Image not found for label: {lab.name}")
            continue

        # copier image + écrire label remappé
        shutil.copy2(img_path, out_images / img_path.name)
        remap_and_write_label(lab, out_labels / lab.name)
        kept += 1

    print(f"✅ {split}: kept {kept} images with ball/rim/person/shoot -> {out_split}")


def write_yaml(dataset_dir: Path) -> Path:
    """
    Create a YOLO dataset yaml that points to the new splits.
    We keep paths relative to the yaml (same style as Roboflow).
    """
    out_yaml = dataset_dir / "data_ball_rim_person_shoot.yaml"
    yaml_text = """train: ../train_ball_rim_person_shoot/images
val: ../valid_ball_rim_person_shoot/images
test: ../test_ball_rim_person_shoot/images

nc: 4
names: ['ball', 'rim', 'person', 'shoot']
"""
    out_yaml.write_text(yaml_text, encoding="utf-8")
    return out_yaml


def main():
    dataset_dir = find_dataset_dir()
    print("Dataset selected:", dataset_dir.resolve())

    for split in SPLITS:
        process_split(dataset_dir, split)

    out_yaml = write_yaml(dataset_dir)

    print("\n✅ Done. You can now train with:")
    print(f"  data={out_yaml}")


if __name__ == "__main__":
    main()
