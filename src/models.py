"""Model definitions: improved CNN, ResNet18 and EfficientNet-B0."""

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


class SimpleCNN(nn.Module):
    """Improved custom CNN trained from scratch."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.15),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.20),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=256,
                out_features=128,
            ),

            nn.ReLU(),
            nn.Dropout(p=0.40),

            nn.Linear(
                in_features=128,
                out_features=num_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def build_resnet18(num_classes=NUM_CLASSES):
    """Create pretrained ResNet18 and replace final classifier."""

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features=in_features,
        out_features=num_classes,
    )

    return model


def build_efficientnet_b0(num_classes=NUM_CLASSES):
    """Create pretrained EfficientNet-B0 and replace final classifier."""

    weights = models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features=in_features,
        out_features=num_classes,
    )

    return model