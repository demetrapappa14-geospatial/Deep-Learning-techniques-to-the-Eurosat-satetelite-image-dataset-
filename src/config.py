"""Central configuration for the EuroSAT deep learning project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_DATASET_DIR = PROJECT_ROOT / "yolo_pseudo_dataset"

for folder in [DATA_DIR, RESULTS_DIR, MODELS_DIR, YOLO_DATASET_DIR]:
    folder.mkdir(exist_ok=True)

NUM_CLASSES = 10
IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.0003
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Windows-safe default. Increase to 2 or 4 only if your PC handles multiprocessing well.
NUM_WORKERS = 0

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
