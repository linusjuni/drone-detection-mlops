import torch
import torch.nn as nn

from drone_detector_mlops.model import DroneDetectorModel, get_model


def test_drone_detector_model_is_nn_module():
    """Test that DroneDetectorModel is a valid PyTorch nn.Module."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    assert isinstance(model, nn.Module)


def test_drone_detector_model_forward_shape():
    """Test that model output has correct shape."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()

    # Create dummy input: batch of 4 RGB images of size 224x224
    x = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    # Output should be (batch_size, num_classes)
    assert output.shape == (4, 2)


def test_drone_detector_model_custom_num_classes():
    """Test that model respects num_classes parameter."""
    model = DroneDetectorModel(num_classes=5, pretrained=False)
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 5)


def test_get_model_factory():
    """Test that get_model factory function returns DroneDetectorModel."""
    model = get_model(num_classes=2, pretrained=False)
    assert isinstance(model, DroneDetectorModel)


def test_get_model_default_args():
    """Test that get_model works with default arguments."""
    model = get_model(pretrained=False)
    assert isinstance(model, DroneDetectorModel)
