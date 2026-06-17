"""Training, evaluation, metrics and visualization utilities."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from config import EUROSAT_CLASSES
from config import MODELS_DIR
from config import RESULTS_DIR


def get_device():
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def evaluate_one_epoch(model, dataloader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_name):
    """Train a model and save the best validation checkpoint."""
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_accuracy = 0.0
    best_model_path = RESULTS_DIR / f"{model_name}_best.pth"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best checkpoint: {best_model_path}")

    save_history(history, model_name)
    plot_training_curves(history, model_name)

    return history, best_model_path


def collect_predictions(model, dataloader, device):
    """Collect all true labels and predictions."""
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting predictions", leave=False):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_predictions)


def evaluate_on_test_set(model, test_loader, device, model_name):
    """Evaluate on the test set and save metrics, report and confusion matrices."""
    labels, predictions = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        target_names=EUROSAT_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(labels, predictions)

    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / f"{model_name}_classification_report.csv"
    )

    plot_confusion_matrix(matrix, model_name, normalized=False)
    plot_confusion_matrix(matrix, model_name, normalized=True)

    metrics = {
        "model": model_name,
        "test_accuracy": float(accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }

    with open(RESULTS_DIR / f"{model_name}_test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\n{model_name} Test Accuracy: {accuracy:.4f}")
    print(f"{model_name} Macro F1:      {metrics['macro_f1']:.4f}")

    return metrics


def save_history(history, model_name):
    """Save training history as JSON."""
    with open(RESULTS_DIR / f"{model_name}_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def plot_training_curves(history, model_name):
    """Save loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_accuracy_curve.png", dpi=300)
    plt.close()


def plot_confusion_matrix(matrix, model_name, normalized=False):
    """Save confusion matrix as an image."""
    if normalized:
        matrix_to_plot = matrix.astype(float) / matrix.sum(axis=1, keepdims=True).clip(min=1)
        filename = f"{model_name}_confusion_matrix_normalized.png"
        title = f"{model_name} Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        matrix_to_plot = matrix
        filename = f"{model_name}_confusion_matrix.png"
        title = f"{model_name} Confusion Matrix"
        fmt = "d"

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix_to_plot, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(EUROSAT_CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(EUROSAT_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(EUROSAT_CLASSES)

    threshold = matrix_to_plot.max() / 2 if matrix_to_plot.max() > 0 else 0.5

    for i in range(matrix_to_plot.shape[0]):
        for j in range(matrix_to_plot.shape[1]):
            value = matrix_to_plot[i, j]
            text = format(value, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close()


def save_final_model(model, model_name):
    """Save final best model weights into models/ for inference."""
    final_path = MODELS_DIR / f"{model_name}.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to: {final_path}")
    return final_path
