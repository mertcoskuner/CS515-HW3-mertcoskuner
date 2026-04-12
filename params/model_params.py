"""Model architecture configuration dataclasses and builders."""

from dataclasses import dataclass, field
from typing import Any, List, Union
from torchvision.models import ResNet18_Weights


@dataclass
class SimpleCNNParams:
    """Parameters for the SimpleCNN architecture.

    Attributes:
        model: Model identifier used by build_model.
        num_classes: Number of output classes.
    """

    model: str = "SimpleCNN"
    num_classes: int = 10


@dataclass
class MobileNetParams:
    """Parameters for the MobileNetV2 architecture.

    Attributes:
        model: Model identifier used by build_model.
        num_classes: Number of output classes.
    """

    model: str = "MobileNet"
    num_classes: int = 10


@dataclass
class ResnetParams:
    """Parameters for the ResNet architecture.

    Attributes:
        model: Model identifier used by build_model.
        resnet_layers: Number of blocks per layer group (e.g. [2,2,2,2] for ResNet-18).
        num_classes: Number of output classes.
    """

    model: str = "ResNet"
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_classes: int = 10


@dataclass
class VGGParams:
    """Parameters for the VGG architecture.

    Attributes:
        model: Model identifier used by build_model.
        vgg_depth: VGG depth variant. One of '11', '13', '16', '19'.
        num_classes: Number of output classes.
    """

    model: str = "VGG"
    vgg_depth: str = "16"
    num_classes: int = 10


@dataclass
class PartAParams:
    """Parameters for PART_A transfer learning with pretrained ResNet-18.

    Attributes:
        model: Model identifier used by build_model.
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        weights: Specific pretrained weight set to load. None if pretrained=False.
        option: Adaptation strategy. 'resize' freezes early layers and upsamples
            inputs to 224x224; 'modify' changes conv1+maxpool for 32x32 inputs
            and fine-tunes all layers.
        input_size: Spatial size of input images fed to the model.
    """

    model: str = "PartA"
    num_classes: int = 10
    pretrained: bool = True
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT
    option: str = "resize"
    input_size: int = 224


def get_simplecnn_params(args: Any) -> SimpleCNNParams:
    """Build SimpleCNNParams from parsed command-line arguments.

    Args:
        args: Parsed argument namespace containing 'num_classes'.

    Returns:
        SimpleCNNParams populated with values from args.
    """
    return SimpleCNNParams(
        num_classes=args.num_classes,
    )


def get_mobilenet_params(args: Any) -> MobileNetParams:
    """Build MobileNetParams from parsed command-line arguments.

    Args:
        args: Parsed argument namespace containing 'num_classes'.

    Returns:
        MobileNetParams populated with values from args.
    """
    return MobileNetParams(
        num_classes=args.num_classes,
    )


def get_resnet_params(args: Any) -> ResnetParams:
    """Build ResnetParams from parsed command-line arguments.

    Args:
        args: Parsed argument namespace containing 'resnet_layers' and 'num_classes'.

    Returns:
        ResnetParams populated with values from args.
    """
    return ResnetParams(
        resnet_layers=args.resnet_layers,
        num_classes=args.num_classes,
    )


def get_vgg_params(args: Any) -> VGGParams:
    """Build VGGParams from parsed command-line arguments.

    Args:
        args: Parsed argument namespace containing 'vgg_depth' and 'num_classes'.

    Returns:
        VGGParams populated with values from args.
    """
    return VGGParams(
        vgg_depth=args.vgg_depth,
        num_classes=args.num_classes,
    )


def get_parta_params(args: Any) -> PartAParams:
    """Build PartAParams from parsed command-line arguments.

    Args:
        args: Parsed argument namespace containing 'num_classes' and 'pretrained'.

    Returns:
        PartAParams populated with values from args. option and input_size use
        their defaults as main.py always overrides them for both options via replace().
    """
    return PartAParams(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        weights=ResNet18_Weights.DEFAULT if args.pretrained else None,
    )


ModelParams = Union[SimpleCNNParams, MobileNetParams, ResnetParams, VGGParams, PartAParams]


def get_model_params(args: Any) -> ModelParams:
    """Build and return the model parameter object for the chosen model.

    Args:
        args: Parsed argument namespace containing at least a 'model' field.

    Returns:
        A model-specific params dataclass instance.

    Raises:
        ValueError: If the model name is not recognised.
    """
    model_builders = {
        "SIMPLECNN": get_simplecnn_params,
        "MOBILENET": get_mobilenet_params,
        "RESNET": get_resnet_params,
        "VGG": get_vgg_params,
        "PARTA": get_parta_params,
    }
    model_key = args.model.upper()
    if model_key not in model_builders:
        raise ValueError(
            f"Unknown model name: {args.model}. "
            f"Available models: {', '.join(model_builders.keys())}"
        )
    return model_builders[model_key](args)
