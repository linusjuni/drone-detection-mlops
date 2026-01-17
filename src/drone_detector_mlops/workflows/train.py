import torch
import wandb
import time
import hydra
from omegaconf import DictConfig, OmegaConf
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
timestamp = time.strftime("%Y%m%d-%H%M%S")


@hydra.main(version_base=None, config_path="/app/configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Run training with Hydra config."""
    logger.info("Training config:\n" + OmegaConf.to_yaml(cfg))

    epochs = cfg.hyper_parameters.epochs
    batch_size = cfg.hyper_parameters.batch_size
    lr = cfg.hyper_parameters.lr

    logger.info(
        "Starting training",
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
        name=f"resnet18-lr{lr}-bs{batch_size}-e{epochs}-{timestamp}",
        tags=["sweep", "optuna"] if cfg.get("multirun", False) else ["single-run"],
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

    # Initialize storage
    storage = get_storage()

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=storage.data_dir,
        splits_dir=storage.splits_dir,
        batch_size=batch_size,
        num_workers=4,
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
            # Include hyperparameters in filename for sweep tracking
            model_filename = f"model-lr{lr}-bs{batch_size}-e{epochs}-{timestamp}"

            model_path = storage.save_model(model, model_filename)
            logger.success("Best model saved", path=str(model_path), val_loss=best_val_loss)

            # Log model artifacts to W&B
            artifact = wandb.Artifact(
                name=f"model-lr{lr}-bs{batch_size}-e{epochs}-{timestamp}",
                type="model",
                description=f"ResNet18 drone detector (val_loss={best_val_loss:.4f}, lr={lr}, bs={batch_size}, epochs={epochs})",
                metadata={
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "val_loss": best_val_loss,
                    "val_acc": val_metrics["accuracy"],
                    "architecture": "resnet18",
                },
            )

            # Add both PyTorch and ONNX models
            if storage.mode == "local":
                artifact.add_file(str(model_path))  # .pth file
                artifact.add_file(str(model_path).replace(".pth", ".onnx"))  # .onnx file
                # Skip model-latest during sweeps to avoid conflicts between trials
                if not cfg.get("multirun", False):
                    artifact.add_file("models/model-latest.pth")
                    artifact.add_file("models/model-latest.onnx")
            else:
                # For cloud mode, just log the GCS paths
                artifact.add_reference(str(model_path))
                artifact.add_reference(str(model_path).replace(".pth", ".onnx"))

            wandb.log_artifact(artifact)

    wandb.finish()
    logger.success(
        "Training completed",
        best_val_loss=best_val_loss,
        hyperparameters={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    return val_metrics["loss"]  # Return for sweeper optimization


if __name__ == "__main__":
    main()
