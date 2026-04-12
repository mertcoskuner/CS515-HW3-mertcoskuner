import torch
import torch.nn as nn
from typing import List, Union


class VGG(nn.Module):
    """VGG convolutional neural network for image classification.

    Implements VGG-11, 13, 16, and 19 configurations with batch normalisation.
    Supports CIFAR-10 via the 'dept' depth selector.

    Args:
        dept: VGG variant depth key. One of '11', '13', '16', '19'.
        norm: Normalisation layer class applied after each convolution. Default: nn.BatchNorm2d.
        num_classes: Number of output classes. Default: 10.
    """

    def __init__(self, dept: str, norm: type = nn.BatchNorm2d, num_classes: int = 10) -> None:
        super().__init__()
        self.features = self.make_layers_vgg(dept, norm)

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and classifier.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Class logits of shape (N, num_classes).
        """
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

    def make_layers_vgg(self, dept: str, norm: type = nn.BatchNorm2d) -> nn.Sequential:
        """Build the feature-extraction block from a VGG configuration string.

        Args:
            dept: VGG depth key selecting the layer config. One of '11', '13', '16', '19'.
            norm: Normalisation layer class inserted after each Conv2d. Default: nn.BatchNorm2d.

        Returns:
            nn.Sequential containing the Conv/BN/ReLU/MaxPool stack.
        """
        layers: List[nn.Module] = []
        in_channels = 3
        cfg = {
            '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        for v in cfg[dept]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, norm(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
