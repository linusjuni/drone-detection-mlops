import torch
from torch import nn
from torch.utils.data import DataLoader

from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class metrics
    class_correct = [0, 0]  # [drone, bird]
    class_total = [0, 0]    # [drone, bird]

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Calculate metrics
    overall_accuracy = correct / total
    drone_accuracy = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    bird_accuracy = class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": overall_accuracy,
        "drone_accuracy": drone_accuracy,
        "bird_accuracy": bird_accuracy,
    }
