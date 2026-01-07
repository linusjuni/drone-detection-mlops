from pathlib import Path
import torch
from torch import nn
import typer

from drone_detector_mlops.data.transforms import test_transform
from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.data.data import get_dataloaders
from drone_detector_mlops.model import get_model
from drone_detector_mlops.workflows.testing import evaluate_model

logger = get_logger(__name__)
app = typer.Typer()


@app.command()
def main(
    checkpoint: Path = "models/model.pth",
    data_dir: Path = "data",
    batch_size: int = 32,
):
    data_dir = Path(data_dir)

    logger.info("Starting evaluation", checkpoint=str(checkpoint))

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Using device", device=str(device))

    # Load model
    model = get_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logger.success("Model loaded successfully")

    # Load test data WITH TRANSFORMS
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        splits_dir=data_dir / "splits",
        batch_size=batch_size,
        transforms_dict={"test": test_transform},  # ADD THIS
    )

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, criterion, device)

    # Log results
    logger.info(
        "Evaluation complete",
        test_loss=f"{metrics['loss']:.4f}",
        test_accuracy=f"{metrics['accuracy']:.4f}",
        drone_accuracy=f"{metrics['drone_accuracy']:.4f}",
        bird_accuracy=f"{metrics['bird_accuracy']:.4f}",
    )

    logger.success(
        f"Test Accuracy: {metrics['accuracy'] * 100:.2f}% "
        f"(Drone: {metrics['drone_accuracy'] * 100:.2f}%, Bird: {metrics['bird_accuracy'] * 100:.2f}%)"
    )


if __name__ == "__main__":
    app()
