import torch
import torch.nn as nn
import pytest

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


def test_drone_detector_model_pretrained_parameter():
    """Test that pretrained parameter is respected."""
    # This mainly tests that it doesn't crash - actual weight values are hard to verify
    model_pretrained = DroneDetectorModel(num_classes=2, pretrained=True)
    model_not_pretrained = DroneDetectorModel(num_classes=2, pretrained=False)

    assert isinstance(model_pretrained, DroneDetectorModel)
    assert isinstance(model_not_pretrained, DroneDetectorModel)


def test_drone_detector_model_output_is_logits():
    """Test that model output is raw logits (not probabilities)."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    # Logits should NOT sum to 1 across classes (which would indicate softmax)
    row_sums = output.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_drone_detector_model_gradient_flow():
    """Test that gradients flow through the model (can be trained)."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.train()

    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1])

    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)

    # Backward pass
    loss.backward()

    # Check that at least some parameters have gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_gradients


def test_drone_detector_model_train_eval_modes():
    """Test that model can switch between train and eval modes."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)

    # Test train mode
    model.train()
    assert model.training

    # Test eval mode
    model.eval()
    assert not model.training


def test_drone_detector_model_different_batch_sizes():
    """Test that model handles different batch sizes."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()

    batch_sizes = [1, 2, 8, 16, 32]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 2)


def test_drone_detector_model_has_parameters():
    """Test that model has trainable parameters."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)

    params = list(model.parameters())
    assert len(params) > 0

    # Check that at least some parameters require grad
    trainable_params = [p for p in params if p.requires_grad]
    assert len(trainable_params) > 0


def test_drone_detector_model_parameter_count():
    """Test that model has expected number of parameters (ResNet18 size)."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)

    total_params = sum(p.numel() for p in model.parameters())

    # ResNet18 has roughly 11-12 million parameters
    # Allow some variation for final layer differences
    assert 10_000_000 < total_params < 15_000_000


def test_get_model_with_pretrained_true():
    """Test that get_model can create pretrained model."""
    model = get_model(num_classes=2, pretrained=True)
    assert isinstance(model, DroneDetectorModel)


def test_drone_detector_model_output_dtype():
    """Test that model output is float tensor."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.dtype == torch.float32


def test_drone_detector_model_device_transfer():
    """Test that model can be moved to different devices."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)

    # Test CPU
    model_cpu = model.to("cpu")
    assert next(model_cpu.parameters()).device.type == "cpu"

    # Only test CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        assert next(model_cuda.parameters()).device.type == "cuda"


def test_drone_detector_model_wrong_input_shape_raises_error():
    """Test that model raises error for incorrect input shape."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()

    # Wrong number of channels (should be 3 for RGB)
    x_wrong_channels = torch.randn(1, 1, 224, 224)

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(x_wrong_channels)


def test_get_model_creates_new_instance():
    """Test that get_model creates new independent instances."""
    model1 = get_model(num_classes=2, pretrained=False)
    model2 = get_model(num_classes=2, pretrained=False)

    # Should be different objects
    assert model1 is not model2

    # But same type
    assert type(model1) is type(model2)
