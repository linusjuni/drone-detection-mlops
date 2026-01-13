from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import gcsfs

from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class DroneVsBirdDataset(Dataset):
    """Dataset that works with both local and GCS paths."""

    def __init__(self, data_dir, split_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_map = {"drone": 0, "bird": 1}
        self._gcsfs = None

        # Read split file (handles both local and GCS)
        self.image_paths = self._read_split_file(split_file)
        self.labels = [self.label_map[Path(p).parent.name] for p in self.image_paths]

    @property
    def gcsfs(self):
        """Lazy-load gcsfs filesystem."""
        if self._gcsfs is None:
            self._gcsfs = gcsfs.GCSFileSystem()
        return self._gcsfs

    def _is_gcs_path(self, path) -> bool:
        """Check if path is a GCS path."""
        return str(path).startswith("gs://")

    def _read_split_file(self, split_file) -> list:
        """Read split file from local disk or GCS."""
        if self._is_gcs_path(split_file):
            with self.gcsfs.open(str(split_file), "r") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            with open(split_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

    def _build_image_path(self, idx):
        """Build full image path (local or GCS)."""
        rel_path = self.image_paths[idx]
        if self._is_gcs_path(self.data_dir):
            return f"{self.data_dir}/{rel_path}"
        else:
            return Path(self.data_dir) / rel_path

    def _open_image(self, img_path):
        """Open image from local disk or GCS."""
        if self._is_gcs_path(img_path):
            with self.gcsfs.open(str(img_path), "rb") as f:
                return Image.open(f).convert("RGB")
        else:
            return Image.open(img_path).convert("RGB")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self._build_image_path(idx)
        image = self._open_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir,
    splits_dir,
    batch_size: int,
    num_workers: int = 0,
    transforms_dict=None,
):
    """Create dataloaders for train/val/test splits."""
    transforms_dict = transforms_dict or {}

    datasets = {}
    for split in ["train", "val", "test"]:
        # Handle both local Path and GCS string paths
        if isinstance(splits_dir, Path):
            split_file = splits_dir / f"{split}_files.txt"
        else:
            split_file = f"{splits_dir}/{split}_files.txt"

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
