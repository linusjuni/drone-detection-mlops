from torch.utils.data import Dataset

from drone_detector_mlops.data.data import DroneVsBirdDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = DroneVsBirdDataset("data/raw")
    assert isinstance(dataset, Dataset)
