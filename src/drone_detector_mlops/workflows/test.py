import argparse
from pathlib import Path
import torch
from torch import nn

from drone_detector_mlops.data.transforms import test_transform
from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.data.data import get_dataloaders
from drone_detector_mlops.model import get_model
from drone_detector_mlops.workflows.testing import evaluate_model

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate drone detector model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=Path, default="data", help="Path to data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    logger.info("Starting evaluation", checkpoint=str(args.checkpoint))

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Using device", device=str(device))

    # Load model
    model = get_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.success("Model loaded successfully")

    # Load test data WITH TRANSFORMS
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        splits_dir=args.data_dir / "splits",
        batch_size=args.batch_size,
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
    main()
