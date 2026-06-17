"""Train YOLOv8 on automatically generated pseudo EuroSAT bounding boxes."""

from ultralytics import YOLO

from config import PROJECT_ROOT
from config import YOLO_DATASET_DIR


def main():
    data_yaml = YOLO_DATASET_DIR / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            "YOLO pseudo dataset not found. Run: python yolo_create_pseudo_dataset.py"
        )

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=224,
        batch=16,
        project=str(PROJECT_ROOT / "results" / "yolo_runs"),
        name="eurosat_yolo_pseudo_detection",
        exist_ok=True,
    )

    print("\nYOLO training finished.")
    print(f"Results: {PROJECT_ROOT / 'results' / 'yolo_runs'}")


if __name__ == "__main__":
    main()
