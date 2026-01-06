import timm
import torch
import torch.nn as nn


class DroneDetectorModel(nn.Module):
    """Drone vs Bidfsdfrd classifier usinfffg TIMM ResNet18."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet18 from TIMM
        self.model = timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model(num_classes: int = 2, pretrained: bool = True) -> DroneDetectorModel:
    """Factory function to create model."""
    return DroneDetectorModel(num_classes=num_classes, pretrained=pretrained)
