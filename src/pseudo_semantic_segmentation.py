"""Pseudo semantic segmentation visualization for EuroSAT.

EuroSAT has no official pixel masks. This script creates heuristic masks
for visualization and analysis only.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import EuroSAT
from torchvision import transforms

from config import DATA_DIR
from config import EUROSAT_CLASSES
from config import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "pseudo_semantic_segmentation"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_mask(image_np, class_name):
    """Create a heuristic class-dependent mask."""
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


def main():
    dataset = EuroSAT(
        root=str(DATA_DIR),
        download=True,
        transform=transforms.ToTensor(),
    )

    examples = {}

    for image, label in dataset:
        class_name = dataset.classes[label]

        if class_name not in examples:
            examples[class_name] = image

        if len(examples) == len(EUROSAT_CLASSES):
            break

    for class_name in EUROSAT_CLASSES:
        image = examples[class_name]
        image_np = image.permute(1, 2, 0).numpy()
        mask = create_mask(image_np, class_name)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Pseudo Semantic Segmentation: {class_name}", fontsize=18, fontweight="bold")

        axes[0].imshow(image_np, interpolation="nearest")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="turbo", interpolation="nearest")
        axes[1].set_title("Pseudo Mask")
        axes[1].axis("off")

        axes[2].imshow(image_np, interpolation="nearest")
        axes[2].imshow(mask, cmap="turbo", alpha=0.45, interpolation="nearest")
        axes[2].set_title("Transparent Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"{class_name}_pseudo_segmentation.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
