"""Download EuroSAT into data/ and print basic dataset information."""

from torchvision import datasets
from torchvision import transforms

from config import DATA_DIR


def main():
    dataset = datasets.EuroSAT(
        root=str(DATA_DIR),
        download=True,
        transform=transforms.ToTensor(),
    )

    print("\nEuroSAT dataset is ready.")
    print(f"Dataset folder: {DATA_DIR}")
    print(f"Number of images: {len(dataset)}")
    print(f"Classes: {dataset.classes}\n")


if __name__ == "__main__":
    main()
