from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from drone_detector_mlops.data.data import DroneVsBirdDataset, get_dataloaders
from drone_detector_mlops.data.create_splits import create_splits, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from drone_detector_mlops.data.transforms import (
    train_transform,
    val_transform,
    test_transform,
)


def test_drone_vs_bird_dataset_is_dataset(tmp_path):
    """Test that DroneVsBirdDataset is a valid PyTorch Dataset."""
    # Create test directory structure
    data_dir = tmp_path / "data"
    drone_dir = data_dir / "drone"
    bird_dir = data_dir / "bird"
    drone_dir.mkdir(parents=True)
    bird_dir.mkdir(parents=True)

    # Create dummy image files
    (drone_dir / "drone1.jpg").touch()
    (bird_dir / "bird1.jpg").touch()

    # Create split file
    split_file = tmp_path / "split.txt"
    split_file.write_text("drone/drone1.jpg\nbird/bird1.jpg\n")

    # Mock Image.open to avoid needing real images
    with patch("drone_detector_mlops.data.data.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_open.return_value = mock_img

        dataset = DroneVsBirdDataset(data_dir, split_file)
        assert isinstance(dataset, Dataset)


def test_drone_vs_bird_dataset_length(tmp_path):
    """Test that dataset length matches number of entries in split file."""
    data_dir = tmp_path / "data"
    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)

    # Create split file with 3 entries
    split_file = tmp_path / "split.txt"
    split_file.write_text("drone/img1.jpg\ndrone/img2.jpg\ndrone/img3.jpg\n")

    dataset = DroneVsBirdDataset(data_dir, split_file)
    assert len(dataset) == 3


def test_drone_vs_bird_dataset_labels(tmp_path):
    """Test that labels are correctly assigned based on directory name."""
    data_dir = tmp_path / "data"
    (data_dir / "drone").mkdir(parents=True)
    (data_dir / "bird").mkdir(parents=True)

    split_file = tmp_path / "split.txt"
    split_file.write_text("drone/img1.jpg\nbird/img2.jpg\ndrone/img3.jpg\n")

    dataset = DroneVsBirdDataset(data_dir, split_file)

    # drone -> 0, bird -> 1
    assert dataset.labels == [0, 1, 0]


def test_drone_vs_bird_dataset_getitem(tmp_path):
    """Test that __getitem__ returns correct image and label."""
    data_dir = tmp_path / "data"
    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)

    split_file = tmp_path / "split.txt"
    split_file.write_text("drone/img1.jpg\n")

    # Mock PIL Image
    with patch("drone_detector_mlops.data.data.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_open.return_value = mock_img

        dataset = DroneVsBirdDataset(data_dir, split_file)
        image, label = dataset[0]

        # Verify correct label
        assert label == 0  # drone
        # Verify image was opened and converted
        mock_open.assert_called_once()
        mock_img.convert.assert_called_once_with("RGB")


def test_drone_vs_bird_dataset_getitem_with_transform(tmp_path):
    """Test that transforms are applied when provided."""
    data_dir = tmp_path / "data"
    bird_dir = data_dir / "bird"
    bird_dir.mkdir(parents=True)

    split_file = tmp_path / "split.txt"
    split_file.write_text("bird/img1.jpg\n")

    # Create a simple transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    with patch("drone_detector_mlops.data.data.Image.open") as mock_open:
        # Create a mock PIL Image
        from PIL import Image

        mock_img = Image.new("RGB", (100, 100))
        mock_open.return_value = mock_img

        dataset = DroneVsBirdDataset(data_dir, split_file, transform=transform)
        image, label = dataset[0]

        # Verify image is now a tensor
        assert isinstance(image, torch.Tensor)
        assert label == 1  # bird


def test_drone_vs_bird_dataset_multiple_items(tmp_path):
    """Test accessing multiple items from dataset."""
    data_dir = tmp_path / "data"
    (data_dir / "drone").mkdir(parents=True)
    (data_dir / "bird").mkdir(parents=True)

    split_file = tmp_path / "split.txt"
    split_file.write_text("drone/img1.jpg\nbird/img2.jpg\ndrone/img3.jpg\n")

    with patch("drone_detector_mlops.data.data.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_open.return_value = mock_img

        dataset = DroneVsBirdDataset(data_dir, split_file)

        # Test accessing all items
        _, label0 = dataset[0]
        _, label1 = dataset[1]
        _, label2 = dataset[2]

        assert label0 == 0  # drone
        assert label1 == 1  # bird
        assert label2 == 0  # drone


def test_drone_vs_bird_dataset_empty_split_file(tmp_path):
    """Test behavior with empty split file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    split_file = tmp_path / "split.txt"
    split_file.write_text("")  # Empty file

    dataset = DroneVsBirdDataset(data_dir, split_file)
    assert len(dataset) == 0


def test_drone_vs_bird_dataset_whitespace_handling(tmp_path):
    """Test that whitespace in split file is handled correctly."""
    data_dir = tmp_path / "data"
    (data_dir / "drone").mkdir(parents=True)

    split_file = tmp_path / "split.txt"
    split_file.write_text("drone/img1.jpg\n\n  \ndrone/img2.jpg\n")  # Empty lines and whitespace

    dataset = DroneVsBirdDataset(data_dir, split_file)
    assert len(dataset) == 2  # Should skip empty/whitespace lines


def test_get_dataloaders_returns_three_loaders(tmp_path):
    """Test that get_dataloaders returns train, val, and test loaders."""
    data_dir = tmp_path / "data"
    (data_dir / "drone").mkdir(parents=True)
    (data_dir / "bird").mkdir(parents=True)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    # Create split files
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}_files.txt"
        split_file.write_text("drone/img1.jpg\nbird/img2.jpg\n")

    with patch("drone_detector_mlops.data.data.Image.open"):
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            splits_dir=splits_dir,
            batch_size=2,
            num_workers=0,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None


def test_get_dataloaders_batch_size(tmp_path):
    """Test that dataloaders use correct batch size."""
    data_dir = tmp_path / "data"
    (data_dir / "drone").mkdir(parents=True)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    # Create split files with 4 samples
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}_files.txt"
        split_file.write_text("drone/img1.jpg\ndrone/img2.jpg\ndrone/img3.jpg\ndrone/img4.jpg\n")

    with patch("drone_detector_mlops.data.data.Image.open") as mock_open:
        from PIL import Image

        mock_img = Image.new("RGB", (100, 100))
        mock_open.return_value = mock_img

        train_loader, _, _ = get_dataloaders(
            data_dir=data_dir,
            splits_dir=splits_dir,
            batch_size=2,
            num_workers=0,
        )

        assert train_loader.batch_size == 2


def test_get_dataloaders_train_shuffle(tmp_path):
    """Test that train loader shuffles but val/test don't."""
    from torch.utils.data.sampler import RandomSampler, SequentialSampler

    data_dir = tmp_path / "data"
    (data_dir / "drone").mkdir(parents=True)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}_files.txt"
        split_file.write_text("drone/img1.jpg\n")

    with patch("drone_detector_mlops.data.data.Image.open"):
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            splits_dir=splits_dir,
            batch_size=1,
            num_workers=0,
        )

        # Train should use RandomSampler (shuffle=True)
        assert isinstance(train_loader.sampler, RandomSampler)

        # Val and test should use SequentialSampler (shuffle=False)
        assert isinstance(val_loader.sampler, SequentialSampler)
        assert isinstance(test_loader.sampler, SequentialSampler)


def test_create_splits_creates_output_dir(tmp_path):
    """Test that output directory is created."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Create dummy data - NEED ENOUGH SAMPLES FOR SPLIT
    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)
    for i in range(10):  # Changed from 1 to 10
        (drone_dir / f"img{i}.jpg").touch()

    create_splits(data_dir, output_dir)

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_create_splits_creates_three_files(tmp_path):
    """Test that train, val, and test split files are created."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Create dummy data
    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)
    for i in range(10):
        (drone_dir / f"img{i}.jpg").touch()

    create_splits(data_dir, output_dir)

    assert (output_dir / "train_files.txt").exists()
    assert (output_dir / "val_files.txt").exists()
    assert (output_dir / "test_files.txt").exists()


def test_create_splits_correct_ratios(tmp_path):
    """Test that split ratios are approximately correct."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Create balanced dataset with 100 images per class
    for class_name in ["drone", "bird"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(100):
            (class_dir / f"img{i}.jpg").touch()

    create_splits(data_dir, output_dir)

    # Read split files
    train_lines = (output_dir / "train_files.txt").read_text().strip().split("\n")
    val_lines = (output_dir / "val_files.txt").read_text().strip().split("\n")
    test_lines = (output_dir / "test_files.txt").read_text().strip().split("\n")

    total = len(train_lines) + len(val_lines) + len(test_lines)

    # Check ratios (allow small tolerance due to rounding)
    assert abs(len(train_lines) / total - TRAIN_RATIO) < 0.02
    assert abs(len(val_lines) / total - VAL_RATIO) < 0.02
    assert abs(len(test_lines) / total - TEST_RATIO) < 0.02


def test_create_splits_stratified(tmp_path):
    """Test that splits are stratified (balanced across classes)."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Create dataset with 50 drones and 50 birds
    for class_name in ["drone", "bird"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(50):
            (class_dir / f"img{i}.jpg").touch()

    create_splits(data_dir, output_dir)

    # Check train split is balanced
    train_lines = (output_dir / "train_files.txt").read_text().strip().split("\n")
    drone_count = sum(1 for line in train_lines if line.startswith("drone"))
    bird_count = sum(1 for line in train_lines if line.startswith("bird"))

    # Should be roughly equal (allow small variation)
    assert abs(drone_count - bird_count) <= 2


def test_create_splits_only_jpg_files(tmp_path):
    """Test that only .jpg/.JPG files are included."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)

    # Create different file types
    (drone_dir / "img1.jpg").touch()
    (drone_dir / "img2.JPG").touch()
    (drone_dir / "img3.jpeg").touch()
    (drone_dir / "img4.JPEG").touch()
    (drone_dir / "img5.png").touch()  # Should be ignored
    (drone_dir / "img6.txt").touch()  # Should be ignored

    create_splits(data_dir, output_dir)

    # Read all split files
    all_lines = []
    for split in ["train", "val", "test"]:
        content = (output_dir / f"{split}_files.txt").read_text().strip()
        if content:
            all_lines.extend(content.split("\n"))

    # Should only have jpg/JPG/jpeg/JPEG files
    assert all(line.lower().endswith((".jpg", ".jpeg")) for line in all_lines)
    assert not any("png" in line or "txt" in line for line in all_lines)


def test_create_splits_relative_paths(tmp_path):
    """Test that paths are relative to data_dir."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)
    for i in range(10):  # Changed from 1 to 10
        (drone_dir / f"img{i}.jpg").touch()

    create_splits(data_dir, output_dir)

    # Read a split file
    train_content = (output_dir / "train_files.txt").read_text().strip()

    # Check first line (should be relative path)
    first_line = train_content.split("\n")[0]
    assert not first_line.startswith("/")
    assert first_line.startswith("drone/")


def test_create_splits_reproducibility(tmp_path):
    """Test that same random_seed produces same splits."""
    data_dir = tmp_path / "data"
    output_dir1 = tmp_path / "splits1"
    output_dir2 = tmp_path / "splits2"

    # Create dataset
    for class_name in ["drone", "bird"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(20):
            (class_dir / f"img{i}.jpg").touch()

    # Run twice
    create_splits(data_dir, output_dir1)
    create_splits(data_dir, output_dir2)

    # Results should be identical
    for split in ["train", "val", "test"]:
        content1 = (output_dir1 / f"{split}_files.txt").read_text()
        content2 = (output_dir2 / f"{split}_files.txt").read_text()
        assert content1 == content2


def test_create_splits_no_data_loss(tmp_path):
    """Test that all images are included in one of the splits."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Create dataset
    total_images = 0
    for class_name in ["drone", "bird"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(25):
            (class_dir / f"img{i}.jpg").touch()
            total_images += 1

    create_splits(data_dir, output_dir)

    # Count total images in splits
    split_total = 0
    for split in ["train", "val", "test"]:
        content = (output_dir / f"{split}_files.txt").read_text().strip()
        if content:
            split_total += len(content.split("\n"))

    assert split_total == total_images


def test_create_splits_ignores_non_directories(tmp_path):
    """Test that non-directory files in data_dir are ignored."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Create valid class directory
    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)
    for i in range(10):  # Changed from 1 to 10
        (drone_dir / f"img{i}.jpg").touch()

    # Create a file in data_dir (not a directory)
    (data_dir / "README.txt").touch()

    # Should not raise error
    create_splits(data_dir, output_dir)

    assert (output_dir / "train_files.txt").exists()


def test_create_splits_empty_class_directory(tmp_path):
    """Test behavior with empty class directory."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Create directory with images
    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)
    for i in range(10):
        (drone_dir / f"img{i}.jpg").touch()

    # Create empty directory
    empty_dir = data_dir / "empty_class"
    empty_dir.mkdir(parents=True)

    # Should work fine (just skip empty directory)
    create_splits(data_dir, output_dir)

    train_content = (output_dir / "train_files.txt").read_text()
    assert "empty_class" not in train_content


def test_create_splits_output_dir_exists(tmp_path):
    """Test that function works when output_dir already exists."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "splits"

    # Pre-create output directory
    output_dir.mkdir(parents=True)

    drone_dir = data_dir / "drone"
    drone_dir.mkdir(parents=True)
    for i in range(10):  # Changed from 1 to 10
        (drone_dir / f"img{i}.jpg").touch()

    # Should not raise error
    create_splits(data_dir, output_dir)

    assert (output_dir / "train_files.txt").exists()


def test_train_transform_output_shape():
    """Test that train_transform produces correct output shape."""
    dummy_img = Image.new("RGB", (300, 300), color="red")
    transformed = train_transform(dummy_img)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)


def test_val_transform_output_shape():
    """Test that val_transform produces correct output shape."""
    dummy_img = Image.new("RGB", (300, 300), color="blue")
    transformed = val_transform(dummy_img)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)


def test_test_transform_is_val_transform():
    """Test that test_transform is identical to val_transform."""
    assert test_transform is val_transform


def test_train_transform_has_augmentations():
    """Test that train_transform includes data augmentation."""
    transform_types = [type(t) for t in train_transform.transforms]

    assert transforms.RandomResizedCrop in transform_types
    assert transforms.RandomHorizontalFlip in transform_types


def test_val_transform_no_augmentations():
    """Test that val_transform does not include random augmentations."""
    transform_types = [type(t) for t in val_transform.transforms]

    assert transforms.RandomResizedCrop not in transform_types
    assert transforms.RandomHorizontalFlip not in transform_types
