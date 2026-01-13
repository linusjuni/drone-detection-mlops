from pathlib import Path
import torch
import typer
import wandb
import time

from drone_detector_mlops.utils.settings import settings
from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.utils.storage import get_storage
from drone_detector_mlops.data.data import get_dataloaders
from drone_detector_mlops.data.transforms import train_transform, val_transform
from drone_detector_mlops.workflows.training import (
    setup_training,
    train_epoch,
    validate_epoch,
)

logger = get_logger(__name__)
app = typer.Typer()
timestamp = time.strftime("%Y%m%d-%H%M%S")


@app.command()
def main(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
):
    storage = get_storage()

    logger.info(
        "Starting training",
        mode=settings.MODE,
        data_dir=str(storage.data_dir),
        models_dir=str(storage.models_dir),
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

    # W&B
    wandb.login(key=settings.WANDB_API_KEY)
    wandb.init(
        project=settings.WANDB_PROJECT_NAME,
        name=f"resnet18-{timestamp}",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "resnet18",
            "dataset": "drone-vs-bird",
            "device": str(device),
            "mode": settings.MODE,
        },
    )
    logger.success("W&B initialized", project=wandb.run.project, run_id=wandb.run.id)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=storage.data_dir,
        splits_dir=storage.splits_dir,
        batch_size=batch_size,
        num_workers=0,
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

        # Log to console
        logger.info(
            f"Epoch {epoch + 1}/{epochs}",
            train_loss=train_metrics["loss"],
            train_acc=train_metrics["accuracy"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
        )

        # Log to W&B
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
            }
        )

    # Save model using storage context
    model_path = storage.save_model(model.state_dict(), f"model-{timestamp}.pth")
    logger.success("Training complete", model_path=str(model_path))

    # Log model as W&B artifact
    artifact = wandb.Artifact(
        name=f"drone-detector-model-{timestamp}",
        type="model",
        description="ResNet18 model for drone vs bird classification",
    )

    # Handle both local Path and GCS string
    if isinstance(model_path, Path):
        artifact.add_file(str(model_path))
    else:
        artifact.add_reference(model_path, name="model.pth")

    wandb.log_artifact(artifact)
    logger.success("Model logged to W&B")

    wandb.finish()


if __name__ == "__main__":
    app()
