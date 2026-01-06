from pathlib import Path
import torch
import typer

from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.data.data import get_dataloaders
from drone_detector_mlops.data.transforms import train_transform, val_transform
from drone_detector_mlops.workflows.training import (
    setup_training,
    train_epoch,
    validate_epoch,
)

logger = get_logger(__name__)
app = typer.Typer()


@app.command()
def main(
    data_dir: Path = "data",
    output_dir: Path = "models",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.001,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    logger.info(
        "Starting training",
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Using device", device=str(device))

    model, optimizer, criterion = setup_training(device, lr)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        splits_dir=data_dir / "splits",
        batch_size=batch_size,
        num_workers=4,
        transforms_dict={
            "train": train_transform,
            "val": val_transform,
            "test": val_transform,
        },
    )

    # Training loop
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch + 1}/{epochs}",
            train_loss=train_metrics["loss"],
            train_acc=train_metrics["accuracy"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
        )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    logger.success("Training complete")


if __name__ == "__main__":
    app()
