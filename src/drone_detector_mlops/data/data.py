from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class DroneVsBirdDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_dir: Path, split_file: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_map = {"drone": 0, "bird": 1}

        with open(split_file, "r") as f:
            self.image_paths = [Path(line.strip()) for line in f if line.strip()]

        self.labels = [self.label_map[path.parent.name] for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.data_dir / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir: Path,
    splits_dir: Path,
    batch_size: int,
    num_workers: int = 4,
    transforms_dict=None,
):
    transforms_dict = transforms_dict or {}

    datasets = {}
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}_files.txt"
        datasets[split] = DroneVsBirdDataset(
            data_dir=data_dir,
            split_file=split_file,
            transform=transforms_dict.get(split),
        )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.success("Dataloaders created successfully")

    return train_loader, val_loader, test_loader
