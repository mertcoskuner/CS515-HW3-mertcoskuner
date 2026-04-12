import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class SimpleCNN(nn.Module):
    """Two-layer CNN for CIFAR-10 classification with Kaiming (He) initialisation.

    Applies Kaiming normal initialisation to all Conv2d and Linear layers to
    maintain activation variance with ReLU nonlinearities.

    Args:
        num_classes: Number of output classes. Default: 10.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # assuming input images 32x32
        self.fc2 = nn.Linear(128, num_classes)

        # Apply Kaiming initialization
        self._initialize_weights()

        """
        Kaiming initialization is typically applied to convolutional and linear layers when using ReLU activations.
        It helps maintain the variance of activations through the layers, which can lead to better convergence during training.
        For convolutional layers, we use 'fan_in' mode, which considers the number of input units to the layer.
        'fan_out' mode can also be used if you want to consider the number of output units, but 'fan_in' is more common for ReLU.
        In practice, 'fan_in' is often preferred for ReLU activations because it helps prevent the variance from exploding as we go deeper into the network.

        Formula for Kaiming initialization (for ReLU):
        weight ~ N(0, sqrt(2 / fan_in))

        2.0 is the ReLU correction factor. ReLU zeros out ~half of all activations
        (anything negative), which halves the signal variance at each layer.
        Doubling the initial variance compensates for this, keeping the signal stable.

        """

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal initialisation to all Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through two conv+pool stages and two FC layers.

        Args:
            x: Input tensor of shape (N, 3, 32, 32).

        Returns:
            Class logits of shape (N, num_classes).
        """
        # Conv layers + ReLU + MaxPool
        x = F.relu(self.conv1(x))  # 32 - 3 + 2*1 = 32 (padding=1 keeps size)
        x = F.max_pool2d(x, 2)    # 32x32 -> 16x16

        x = F.relu(self.conv2(x))  # 16 - 3 + 2*1 = 16 (padding=1 keeps size)
        x = F.max_pool2d(x, 2)    # 16x16 -> 8x8

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # output logits

        return x
