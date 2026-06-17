"""Predict one EuroSAT image using a trained model."""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import EUROSAT_CLASSES
from config import IMAGE_SIZE
from config import MODELS_DIR
from config import PROJECT_ROOT
from models import SimpleCNN
from models import build_efficientnet_b0
from models import build_resnet18


def build_model(model_name):
    """Create model architecture by name."""
    if model_name == "simple_cnn":
        return SimpleCNN()

    if model_name == "resnet18_transfer":
        return build_resnet18()

    if model_name == "efficientnet_b0_transfer":
        return build_efficientnet_b0()

    raise ValueError(f"Unknown model: {model_name}")


def find_weights(model_name):
    """Find model weights in models/ or results/."""
    candidates = [
        MODELS_DIR / f"{model_name}.pth",
        PROJECT_ROOT / "results" / f"{model_name}_best.pth",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No weights found for {model_name}. Train the model first."
    )


def load_model(model_name, device):
    """Load trained model."""
    model = build_model(model_name)
    weights_path = find_weights(model_name)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded weights from: {weights_path}")

    return model


def find_eurosat_root():
    """Find the local EuroSAT folder."""
    candidate_roots = [
        PROJECT_ROOT / "data" / "eurosat" / "2750",
        PROJECT_ROOT / "data" / "EuroSAT",
        PROJECT_ROOT / "data" / "2750",
    ]

    for root in candidate_roots:
        if root.exists() and any((root / class_name).exists() for class_name in EUROSAT_CLASSES):
            return root

    raise FileNotFoundError(
        "EuroSAT images were not found. Run: python download_eurosat_dataset.py"
    )


def find_random_image(class_name=None):
    """Pick a random image from EuroSAT."""
    root = find_eurosat_root()

    if class_name is None:
        image_paths = list(root.glob("*/*.jpg"))
    else:
        image_paths = list((root / class_name).glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError("No image files found.")

    return random.choice(image_paths)


def predict_image(model, image_path, device):
    """Return predicted class and confidence."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return EUROSAT_CLASSES[predicted.item()], confidence.item() * 100


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18_transfer",
        choices=["simple_cnn", "resnet18_transfer", "efficientnet_b0_transfer"],
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        choices=EUROSAT_CLASSES,
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)

    image_path = Path(args.image) if args.image else find_random_image(args.class_name)

    predicted_class, confidence = predict_image(model, image_path, device)

    print("\nPrediction Result")
    print("=================")
    print(f"Image: {image_path}")
    print(f"Model: {args.model}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
