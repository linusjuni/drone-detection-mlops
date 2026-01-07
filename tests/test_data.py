from unittest.mock import patch, MagicMock

from torch.utils.data import Dataset

from drone_detector_mlops.data.data import DroneVsBirdDataset


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
