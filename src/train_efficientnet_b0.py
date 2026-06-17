"""Train EfficientNet-B0 with ImageNet transfer learning on EuroSAT."""

import torch
import torch.nn as nn
import torch.optim as optim

from config import LEARNING_RATE
from config import NUM_EPOCHS
from dataset_utils import create_dataloaders
from models import build_efficientnet_b0
from training_utils import evaluate_on_test_set
from training_utils import get_device
from training_utils import save_final_model
from training_utils import train_model


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders()

    model = build_efficientnet_b0()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=NUM_EPOCHS,
        model_name="efficientnet_b0_transfer",
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        model_name="efficientnet_b0_transfer",
    )

    save_final_model(model, "efficientnet_b0_transfer")


if __name__ == "__main__":
    main()
