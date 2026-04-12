"""Dataset and device configuration dataclasses and builders."""

from dataclasses import dataclass
from typing import Any, Union


@dataclass
class DeviceConfig:
    """Holds device and batch size settings shared across configs."""

    device: str
    batch_size: int


@dataclass
class Cifar10Params:
    """Normalisation statistics for the CIFAR-10 dataset."""

    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2023, 0.1994, 0.2010)


def get_cifar10_params(args: Any) -> Cifar10Params:
    """Return Cifar10Params with standard normalisation constants.

    Args:
        args: Parsed argument namespace (unused; kept for uniform signature).

    Returns:
        Cifar10Params with default mean and std values.
    """
    return Cifar10Params()


DatasetParams = Union[Cifar10Params]


def get_data_params(args: Any) -> DatasetParams:
    """Build and return the dataset parameter object for the chosen dataset.

    Args:
        args: Parsed argument namespace containing at least 'dataset',
            'device', and 'batch_size' fields.

    Returns:
        A dataset-specific params dataclass instance.

    Raises:
        ValueError: If the dataset name is not recognised.
    """
    dataset_builders = {
        "CIFAR10": get_cifar10_params,
    }
    dataset_key = args.dataset.upper()
    if dataset_key not in dataset_builders:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            f"Available datasets: {', '.join(dataset_builders.keys())}"
        )
    return dataset_builders[dataset_key](args)
