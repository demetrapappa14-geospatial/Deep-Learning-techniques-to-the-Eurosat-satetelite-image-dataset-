"""Create a YOLO-format pseudo object detection dataset from EuroSAT.

This is not official object detection annotation. EuroSAT has no bounding boxes.
Bounding boxes are generated automatically from heuristic image-processing masks.
"""

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from config import EUROSAT_CLASSES
from config import PROJECT_ROOT
from config import RANDOM_SEED
from config import YOLO_DATASET_DIR


random.seed(RANDOM_SEED)


def find_eurosat_root():
    """Find local EuroSAT folder."""
    candidates = [
        PROJECT_ROOT / "data" / "eurosat" / "2750",
        PROJECT_ROOT / "data" / "EuroSAT",
        PROJECT_ROOT / "data" / "2750",
    ]

    for root in candidates:
        if root.exists() and any((root / class_name).exists() for class_name in EUROSAT_CLASSES):
            return root

    raise FileNotFoundError(
        "EuroSAT data not found. Run: python download_eurosat_dataset.py"
    )


def create_detection_mask(image_np, class_name):
    """Create heuristic foreground mask."""
    r = image_np[:, :, 0]
    g = image_np[:, :, 1]
    b = image_np[:, :, 2]
    brightness = (r + g + b) / 3

    if class_name in ["River", "SeaLake"]:
        return (b > r) & (b > g) & (brightness > 0.16)

    if class_name == "Forest":
        return (g > r) & (g > b) & (brightness > 0.10)

    if class_name == "Highway":
        gray = (np.abs(r - g) < 0.08) & (np.abs(g - b) < 0.08)
        return gray & (brightness > 0.30)

    if class_name in ["Residential", "Industrial"]:
        gray = (np.abs(r - g) < 0.12) & (np.abs(g - b) < 0.12)
        return gray & (brightness > 0.25)

    vegetation = (g > r * 0.85) & (g > b * 0.85)
    return vegetation & (brightness > 0.12)


def mask_to_bbox(mask):
    """Convert mask to bounding box."""
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x_min = int(x_indices.min())
    x_max = int(x_indices.max())
    y_min = int(y_indices.min())
    y_max = int(y_indices.max())

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def bbox_to_yolo_format(bbox, image_width, image_height):
    """Convert pixel bbox to YOLO normalized format."""
    x_min, y_min, x_max, y_max = bbox

    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return x_center, y_center, width, height


def choose_split():
    """Random split."""
    value = random.random()

    if value < 0.70:
        return "train"

    if value < 0.85:
        return "val"

    return "test"


def create_folders():
    """Create YOLO folder structure."""
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

    for split in ["train", "val", "test"]:
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def create_yaml_file():
    """Create data.yaml for Ultralytics YOLO."""
    names = "\n".join(
        f"  {index}: {class_name}"
        for index, class_name in enumerate(EUROSAT_CLASSES)
    )

    yaml_text = f"""path: {YOLO_DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names}
"""

    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    return yaml_path


def main():
    eurosat_root = find_eurosat_root()
    create_folders()

    created = 0
    skipped = 0

    for class_id, class_name in enumerate(EUROSAT_CLASSES):
        class_folder = eurosat_root / class_name

        if not class_folder.exists():
            continue

        for image_path in class_folder.glob("*.jpg"):
            image = Image.open(image_path).convert("RGB")
            image_np = np.asarray(image).astype(np.float32) / 255.0

            mask = create_detection_mask(image_np, class_name)
            bbox = mask_to_bbox(mask)

            if bbox is None:
                skipped += 1
                continue

            image_width, image_height = image.size
            yolo_box = bbox_to_yolo_format(bbox, image_width, image_height)

            split = choose_split()
            output_image_name = f"{class_name}_{image_path.name}"
            output_label_name = output_image_name.replace(".jpg", ".txt")

            output_image_path = YOLO_DATASET_DIR / "images" / split / output_image_name
            output_label_path = YOLO_DATASET_DIR / "labels" / split / output_label_name

            shutil.copy2(image_path, output_image_path)

            line = f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
            output_label_path.write_text(line, encoding="utf-8")

            created += 1

    yaml_path = create_yaml_file()

    print("\nYOLO pseudo dataset created.")
    print(f"Source: {eurosat_root}")
    print(f"Dataset: {YOLO_DATASET_DIR}")
    print(f"YAML: {yaml_path}")
    print(f"Labels created: {created}")
    print(f"Images skipped: {skipped}")


if __name__ == "__main__":
    main()
