import argparse
from pathlib import Path
import torch

from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.data.data import get_dataloaders
from drone_detector_mlops.data.transforms import train_transform, val_transform
from drone_detector_mlops.workflows.training import (
    setup_training,
    train_epoch,
    validate_epoch,
)

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="data")
    parser.add_argument("--output-dir", type=Path, default="models")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    logger.info("Starting training", **vars(args))

    # Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Using device", device=str(device))

    model, optimizer, criterion = setup_training(device, args.lr)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        splits_dir=args.data_dir / "splits",
        batch_size=args.batch_size,
        num_workers=4,
        transforms_dict={
            "train": train_transform,
            "val": val_transform,
            "test": val_transform,
        },
    )

    # Training loop
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}",
            train_loss=train_metrics["loss"],
            train_acc=train_metrics["accuracy"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
        )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pth")
    logger.success("Training complete")


if __name__ == "__main__":
    main()
