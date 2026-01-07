from pathlib import Path
import torch
import typer
import wandb
import time

from drone_detector_mlops.utils.settings import settings
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
timestamp = time.strftime("%Y%m%d-%H%M%S")


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
        },
    )
    logger.success("W&B initialized", project=wandb.run.project, run_id=wandb.run.id)

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

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model-{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    logger.success("Training complete", model_path=str(model_path))

    # Log model as W&B artifact
    artifact = wandb.Artifact(
        name=f"drone-detector-model-{timestamp}",
        type="model",
        description="ResNet18 model for drone vs bird classification",
    )
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    logger.success("Model logged to W&B")

    wandb.finish()


if __name__ == "__main__":
    app()
