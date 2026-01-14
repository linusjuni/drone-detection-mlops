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
    best_val_loss = float("inf")
    for epoch in range(epochs):
        logger.info("Starting epoch", epoch=epoch + 1, total=epochs)

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Log to console
        logger.info(
            "Epoch completed",
            epoch=epoch + 1,
            train_loss=train_metrics["loss"],
            train_acc=train_metrics["accuracy"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
        )

        # Log to W&B
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
        )

        # Save best model (only if validation loss improved)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model_filename = f"model-{timestamp}"

            model_path = storage.save_model(model, model_filename)
            logger.success("Best model saved", path=str(model_path), val_loss=best_val_loss)

            # Log model artifacts to W&B
            artifact = wandb.Artifact(
                name=f"model-{timestamp}",
                type="model",
                description=f"ResNet18 drone detector (val_loss={best_val_loss:.4f})",
            )

            # Add both PyTorch and ONNX models
            if storage.mode == "local":
                artifact.add_file(str(model_path))  # .pth file
                artifact.add_file(str(model_path).replace(".pth", ".onnx"))  # .onnx file
                artifact.add_file("models/model-latest.pth")
                artifact.add_file("models/model-latest.onnx")
            else:
                # For cloud mode, just log the GCS paths
                artifact.add_reference(str(model_path))
                artifact.add_reference(str(model_path).replace(".pth", ".onnx"))

            wandb.log_artifact(artifact)

    wandb.finish()
    logger.success("Training completed", best_val_loss=best_val_loss)


if __name__ == "__main__":
    app()
