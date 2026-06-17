"""Dataset loading, preprocessing and deterministic train/val/test splitting."""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms

from config import BATCH_SIZE
from config import DATA_DIR
from config import IMAGE_SIZE
from config import NUM_WORKERS
from config import RANDOM_SEED
from config import TEST_RATIO
from config import TEST_RATIO
from config import TRAIN_RATIO
from config import VAL_RATIO


def set_seed(seed=RANDOM_SEED):
    """Make the experiment more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_transform():
    """Training preprocessing with stronger augmentation for better CNN generalization."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(degrees=15),

        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def get_eval_transform():
    """Validation/test preprocessing without random augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_dataset(transform):
    """Load EuroSAT and download it automatically if it is missing."""
    return datasets.EuroSAT(
        root=str(DATA_DIR),
        download=True,
        transform=transform,
    )


def create_split_indices(total_size):
    """Create deterministic train/validation/test indices."""
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    shuffled_indices = torch.randperm(total_size, generator=generator).tolist()

    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    return train_indices, val_indices, test_indices


def create_dataloaders():
    """Create independent datasets per split so transforms do not overwrite each other."""
    set_seed()

    base_dataset = load_dataset(transform=get_eval_transform())
    train_indices, val_indices, test_indices = create_split_indices(len(base_dataset))

    train_dataset = Subset(load_dataset(transform=get_train_transform()), train_indices)
    val_dataset = Subset(load_dataset(transform=get_eval_transform()), val_indices)
    test_dataset = Subset(load_dataset(transform=get_eval_transform()), test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader
