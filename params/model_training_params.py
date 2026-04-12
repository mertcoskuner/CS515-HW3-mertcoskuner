"""Configuration parameters for MLP model training."""

from dataclasses import dataclass
from typing import Any

from .data_params import DeviceConfig


@dataclass
class ModelTrainingConfig:
    """Holds all hyperparameters needed to run a training loop."""

    device_config: DeviceConfig
    lr: float
    epochs: int
    weight_decay: float
    dataset: str
    data_dir: str
    num_workers: int
    log_interval: int
    save_path: str
    patience: int
    label_smoothing: float
    kd_temperature: float
    kd_alpha: float

    @property
    def device(self) -> str:
        """Return the device string from the nested DeviceConfig."""
        return self.device_config.device


def get_model_training_params(args: Any) -> ModelTrainingConfig:
    """Construct a ModelTrainingConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace containing device, batch_size, lr,
            epochs, weight_decay, dataset, data_dir, num_workers,
            log_interval, and save_path fields.

    Returns:
        ModelTrainingConfig populated with values from args.
    """
    device_config = DeviceConfig(
        device=args.device,
        batch_size=args.batch_size,
    )
    return ModelTrainingConfig(
        device_config=device_config,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        dataset=args.dataset,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        save_path=args.save_path,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
    )
