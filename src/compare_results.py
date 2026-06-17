"""Compare all available trained models using saved test metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR


MODEL_NAMES = [
    "simple_cnn",
    "resnet18_transfer",
    "efficientnet_b0_transfer",
]


def load_metrics(model_name):
    """Load metrics if they exist."""
    metrics_path = RESULTS_DIR / f"{model_name}_test_metrics.json"

    if not metrics_path.exists():
        print(f"Skipping {model_name}: metrics not found.")
        return None

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    rows = []

    for model_name in MODEL_NAMES:
        metrics = load_metrics(model_name)

        if metrics is not None:
            rows.append(metrics)

    if not rows:
        raise FileNotFoundError(
            "No metrics found. Train at least one model first."
        )

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(by="test_accuracy", ascending=False)

    csv_path = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(comparison["model"], comparison["test_accuracy"])
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_accuracy.png", dpi=300)
    plt.close()

    print("\nModel comparison:")
    print(comparison)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {RESULTS_DIR / 'model_comparison_accuracy.png'}")


if __name__ == "__main__":
    main()
