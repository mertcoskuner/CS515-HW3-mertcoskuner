import random
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple
from torchvision import models
from models.CNN import SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from params.model_params import PartAParams


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all relevant libraries.

    Args:
        seed: Integer seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_params: Any) -> nn.Module:
    """Instantiate and return the model specified by model_params.model.

    Dispatches on the string field model_params.model (case-insensitive) to
    construct the appropriate architecture using the remaining params fields.

    Args:
        model_params: A model-specific params dataclass instance containing at
            least a 'model' string and a 'num_classes' integer field.

    Returns:
        Instantiated nn.Module ready for training.

    Raises:
        ValueError: If model_params.model is not a recognised model name.
    """
    model_name = model_params.model.lower()
    nc = model_params.num_classes

    if model_name == "simplecnn":
        return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        return VGG(dept=model_params.vgg_depth, num_classes=nc)

    if model_name == "resnet":
        return ResNet(BasicBlock, model_params.resnet_layers, num_classes=nc)

    if model_name == "mobilenet":
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def build_parta_model(model_params: PartAParams, device: torch.device) -> nn.Module:
    """Load a fresh pretrained ResNet-18 and adapt it for CIFAR-10 PART_A.

    option='resize': freeze all pretrained layers, replace only FC.
    option='modify': adapt conv1 and maxpool for 32x32 input, fine-tune all layers.

    Args:
        model_params: PartAParams instance with option, weights, and num_classes.
        device: Device to place the model on.

    Returns:
        Adapted ResNet-18 placed on device.
    """
    model = models.resnet18(weights=model_params.weights)

    if model_params.option == "resize":
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, model_params.num_classes)

    elif model_params.option == "modify":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, model_params.num_classes)

    return model.to(device)


def count_flops(model: nn.Module, input_size: Tuple[int, ...], device: torch.device) -> int:
    """Count the number of multiply-accumulate operations for a single forward pass.

    Uses the ptflops library (pip install ptflops) to profile the model.

    Args:
        model: Model to profile.
        input_size: Shape of one input sample without batch dim, e.g. (3, 32, 32).
        device: Device on which to run the profiling pass.

    Returns:
        Total MACs as an integer.
    """
    from ptflops import get_model_complexity_info
    model.eval()
    macs, _ = get_model_complexity_info(
        model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False,
    )
    return int(macs)
