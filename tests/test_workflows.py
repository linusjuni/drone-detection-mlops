import pytest
from unittest.mock import patch, MagicMock
import torch
from torch import nn
from typer.testing import CliRunner
from torch.utils.data import DataLoader, TensorDataset

from drone_detector_mlops.workflows.test import main, app
from drone_detector_mlops.workflows.testing import evaluate_model
from drone_detector_mlops.workflows.training import (
    setup_training,
    train_epoch,
    validate_epoch,
)


@pytest.fixture
def runner():
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_model():
    """Mock PyTorch model."""
    model = MagicMock(spec=nn.Module)
    model.to.return_value = model
    return model


@pytest.fixture
def mock_metrics():
    """Mock evaluation metrics."""
    return {
        "loss": 0.1234,
        "accuracy": 0.9500,
        "drone_accuracy": 0.9600,
        "bird_accuracy": 0.9400,
    }


def test_main_function_runs(mock_model, mock_metrics):
    """Test that main function executes without errors."""
    with (
        patch("drone_detector_mlops.workflows.test.torch.cuda.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.torch.backends.mps.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.get_model", return_value=mock_model),
        patch("drone_detector_mlops.workflows.test.torch.load"),
        patch("drone_detector_mlops.workflows.test.get_dataloaders") as mock_dataloaders,
        patch("drone_detector_mlops.workflows.test.evaluate_model", return_value=mock_metrics),
    ):
        mock_dataloaders.return_value = (None, None, MagicMock())

        main()

        mock_model.load_state_dict.assert_called_once()


def test_main_calls_evaluate_model(mock_model, mock_metrics):
    """Test that evaluate_model is called."""
    with (
        patch("drone_detector_mlops.workflows.test.torch.cuda.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.torch.backends.mps.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.get_model", return_value=mock_model),
        patch("drone_detector_mlops.workflows.test.torch.load"),
        patch("drone_detector_mlops.workflows.test.get_dataloaders") as mock_dataloaders,
        patch("drone_detector_mlops.workflows.test.evaluate_model") as mock_evaluate,
    ):
        mock_dataloaders.return_value = (None, None, MagicMock())
        mock_evaluate.return_value = mock_metrics

        main()

        mock_evaluate.assert_called_once()


def test_cli_runs_successfully(runner, mock_model, mock_metrics):
    """Test that CLI app runs without errors."""
    with (
        patch("drone_detector_mlops.workflows.test.torch.cuda.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.torch.backends.mps.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.get_model", return_value=mock_model),
        patch("drone_detector_mlops.workflows.test.torch.load"),
        patch("drone_detector_mlops.workflows.test.get_dataloaders") as mock_dataloaders,
        patch("drone_detector_mlops.workflows.test.evaluate_model", return_value=mock_metrics),
    ):
        mock_dataloaders.return_value = (None, None, MagicMock())

        result = runner.invoke(app, [])

        assert result.exit_code == 0


def test_evaluate_model_returns_correct_keys(simple_model, simple_dataloader):
    """Test that evaluate_model returns dict with correct keys."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(simple_model, simple_dataloader, criterion, device)

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "drone_accuracy" in metrics
    assert "bird_accuracy" in metrics


def test_evaluate_model_accuracy_values_in_range(simple_model, simple_dataloader):
    """Test that accuracy values are between 0 and 1."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(simple_model, simple_dataloader, criterion, device)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["drone_accuracy"] <= 1
    assert 0 <= metrics["bird_accuracy"] <= 1


def test_evaluate_model_sets_eval_mode(simple_model, simple_dataloader):
    """Test that model is set to eval mode."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    # Set to training mode first
    simple_model.train()
    assert simple_model.training

    evaluate_model(simple_model, simple_dataloader, criterion, device)

    # Should be in eval mode after
    assert not simple_model.training


def test_evaluate_model_loss_is_positive(simple_model, simple_dataloader):
    """Test that loss is a positive number."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(simple_model, simple_dataloader, criterion, device)

    assert metrics["loss"] > 0


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def simple_dataloader():
    """Simple dataloader for testing."""
    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def simple_model(device):
    """Simple model for testing."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2)).to(device)
    return model


# ============================================================================
# Tests for setup_training()
# ============================================================================


def test_setup_training_returns_three_objects(device):
    """Test that setup_training returns model, optimizer, and criterion."""
    model, optimizer, criterion = setup_training(device)

    assert model is not None
    assert optimizer is not None
    assert criterion is not None


def test_setup_training_model_is_nn_module(device):
    """Test that returned model is a PyTorch nn.Module."""
    model, _, _ = setup_training(device)

    assert isinstance(model, nn.Module)


def test_setup_training_optimizer_is_adam(device):
    """Test that optimizer is Adam."""
    _, optimizer, _ = setup_training(device)

    assert isinstance(optimizer, torch.optim.Adam)


def test_setup_training_criterion_is_crossentropy(device):
    """Test that criterion is CrossEntropyLoss."""
    _, _, criterion = setup_training(device)

    assert isinstance(criterion, nn.CrossEntropyLoss)


def test_setup_training_model_on_correct_device(device):
    """Test that model is on the correct device."""
    model, _, _ = setup_training(device)

    # Check first parameter's device
    first_param = next(model.parameters())
    assert first_param.device.type == device.type


def test_setup_training_uses_learning_rate(device):
    """Test that learning rate is applied correctly."""
    _, optimizer, _ = setup_training(device, learning_rate=0.01)

    # Check learning rate in optimizer
    assert optimizer.param_groups[0]["lr"] == 0.01


# ============================================================================
# Tests for train_epoch()
# ============================================================================


def test_train_epoch_returns_correct_keys(simple_model, simple_dataloader, device):
    """Test that train_epoch returns dict with correct keys."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    metrics = train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)

    assert "loss" in metrics
    assert "accuracy" in metrics


def test_train_epoch_accuracy_in_range(simple_model, simple_dataloader, device):
    """Test that accuracy is between 0 and 1."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    metrics = train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)

    assert 0 <= metrics["accuracy"] <= 1


def test_train_epoch_loss_is_positive(simple_model, simple_dataloader, device):
    """Test that loss is positive."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    metrics = train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)

    assert metrics["loss"] > 0


def test_train_epoch_sets_model_to_train_mode(simple_model, simple_dataloader, device):
    """Test that model is set to train mode."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Start in eval mode
    simple_model.eval()
    assert not simple_model.training

    train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)

    # Should be in train mode after
    assert simple_model.training


def test_train_epoch_updates_model_weights(simple_model, simple_dataloader, device):
    """Test that model weights are updated during training."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Get initial weights
    initial_weights = next(simple_model.parameters()).clone()

    train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)

    # Get weights after training
    final_weights = next(simple_model.parameters())

    # Weights should have changed
    assert not torch.equal(initial_weights, final_weights)


# ============================================================================
# Tests for validate_epoch()
# ============================================================================


def test_validate_epoch_returns_correct_keys(simple_model, simple_dataloader, device):
    """Test that validate_epoch returns dict with correct keys."""
    criterion = nn.CrossEntropyLoss()

    metrics = validate_epoch(simple_model, simple_dataloader, criterion, device)

    assert "loss" in metrics
    assert "accuracy" in metrics


def test_validate_epoch_accuracy_in_range(simple_model, simple_dataloader, device):
    """Test that accuracy is between 0 and 1."""
    criterion = nn.CrossEntropyLoss()

    metrics = validate_epoch(simple_model, simple_dataloader, criterion, device)

    assert 0 <= metrics["accuracy"] <= 1


def test_validate_epoch_loss_is_positive(simple_model, simple_dataloader, device):
    """Test that loss is positive."""
    criterion = nn.CrossEntropyLoss()

    metrics = validate_epoch(simple_model, simple_dataloader, criterion, device)

    assert metrics["loss"] > 0


def test_validate_epoch_sets_model_to_eval_mode(simple_model, simple_dataloader, device):
    """Test that model is set to eval mode."""
    criterion = nn.CrossEntropyLoss()

    # Start in train mode
    simple_model.train()
    assert simple_model.training

    validate_epoch(simple_model, simple_dataloader, criterion, device)

    # Should be in eval mode after
    assert not simple_model.training


def test_validate_epoch_does_not_update_weights(simple_model, simple_dataloader, device):
    """Test that model weights are NOT updated during validation."""
    criterion = nn.CrossEntropyLoss()

    # Get initial weights
    initial_weights = next(simple_model.parameters()).clone()

    validate_epoch(simple_model, simple_dataloader, criterion, device)

    # Get weights after validation
    final_weights = next(simple_model.parameters())

    # Weights should NOT have changed
    assert torch.equal(initial_weights, final_weights)


def test_validate_epoch_no_gradient_computation(simple_model, simple_dataloader, device):
    """Test that gradients are not computed during validation."""
    criterion = nn.CrossEntropyLoss()

    validate_epoch(simple_model, simple_dataloader, criterion, device)

    # Check that no gradients were computed
    for param in simple_model.parameters():
        assert param.grad is None
