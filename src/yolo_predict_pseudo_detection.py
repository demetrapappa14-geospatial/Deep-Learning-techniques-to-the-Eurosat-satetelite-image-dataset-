"""Run prediction with the trained YOLOv8 pseudo object detector."""

import argparse
import random
from pathlib import Path

from ultralytics import YOLO

from config import EUROSAT_CLASSES
from config import PROJECT_ROOT


YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "yolo_runs"
    / "eurosat_yolo_pseudo_detection"
    / "weights"
    / "best.pt"
)


def find_eurosat_root():
    """Find EuroSAT image folder."""
    candidates = [
        PROJECT_ROOT / "data" / "eurosat" / "2750",
        PROJECT_ROOT / "data" / "EuroSAT",
        PROJECT_ROOT / "data" / "2750",
    ]

    for root in candidates:
        if root.exists() and any((root / class_name).exists() for class_name in EUROSAT_CLASSES):
            return root

    raise FileNotFoundError("EuroSAT data not found.")


def find_random_image(class_name=None):
    """Pick random image."""
    root = find_eurosat_root()

    if class_name is None:
        image_paths = list(root.glob("*/*.jpg"))
    else:
        image_paths = list((root / class_name).glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError("No images found.")

    return random.choice(image_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None, choices=EUROSAT_CLASSES)
    args = parser.parse_args()

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained YOLO model not found. Run: python train_yolo_pseudo_detection.py"
        )

    image_path = Path(args.image) if args.image else find_random_image(args.class_name)

    model = YOLO(str(YOLO_MODEL_PATH))

    model.predict(
        source=str(image_path),
        save=True,
        project=str(PROJECT_ROOT / "results" / "yolo_predictions"),
        name="predictions",
        exist_ok=True,
        conf=0.25,
    )

    print("\nYOLO prediction completed.")
    print(f"Image: {image_path}")
    print(f"Output: {PROJECT_ROOT / 'results' / 'yolo_predictions' / 'predictions'}")


if __name__ == "__main__":
    main()
