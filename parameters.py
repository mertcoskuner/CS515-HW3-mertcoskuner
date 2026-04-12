"""Command-line argument parsing for the CNN classifier."""

import argparse
import torch


def get_params() -> argparse.Namespace:
    """Define and parse command-line arguments for the CNN classifier.

    Returns:
        argparse.Namespace with all parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CNN Classifier Parameters Setup")
    parser.add_argument("--hw_part", choices=["PART_A", "PART_B", "PART_C"], default="PART_B")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset", choices=["Cifar10"], default="Cifar10")
    parser.add_argument("--model", choices=["SimpleCNN", "MobileNet", "ResNet", "VGG", "PartA"], default="SimpleCNN")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--parta_save_path", type=str, default="results/parta/best_model.pth",
                        help="Base save path used when training PART_A (to load original checkpoints).")
    parser.add_argument("--partb_save_path", type=str, default="results/partb/best_model.pth",
                        help="Base save path used when training PART_B (to load original checkpoints).")
    parser.add_argument("--cifar10c_dir", type=str, default="./data/CIFAR-10-C",
                        help="Path to directory containing CIFAR-10-C .npy files.")
    parser.add_argument("--augmix_severity", type=int, default=3,
                        help="AugMix operation magnitude [1–10].")
    parser.add_argument("--augmix_width", type=int, default=3,
                        help="Number of parallel augmentation chains.")
    parser.add_argument("--augmix_depth", type=int, default=-1,
                        help="Chain length; -1 = random in [1, 3].")
    parser.add_argument("--lambda_jsd", type=float, default=12.0,
                        help="Weight on the JSD consistency loss term.")

    return parser.parse_args()
