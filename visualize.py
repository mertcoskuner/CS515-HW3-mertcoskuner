"""Visualization utilities for CS515 HW2 results.

Generates training-curve plots and summary bar charts for Part A (transfer
learning) and Part B (knowledge distillation) from the JSON result files
produced by train.py.

Usage::

    python visualize.py --hw_part PART_A --results_dir results/parta --plots_dir plots/parta
    python visualize.py --hw_part PART_B --results_dir results/partb --plots_dir plots/partb
    python visualize.py --hw_part ALL   --results_dir results       --plots_dir plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    """Load a JSON result file.

    Args:
        path: Absolute or relative path to the JSON file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path, "r") as f:
        return json.load(f)


def extract_history(
    data: dict,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Extract per-epoch training history from a result dict.

    Args:
        data: Parsed JSON result dictionary containing a ``'history'`` list.

    Returns:
        Tuple of ``(epochs, train_losses, val_losses, train_accs, val_accs)``.
    """
    history = data["history"]
    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]
    train_acc   = [h["train_acc"]  for h in history]
    val_acc     = [h["val_acc"]    for h in history]
    return epochs, train_loss, val_loss, train_acc, val_acc


def save_fig(fig: plt.Figure, path: str) -> None:
    """Save a figure to disk and close it.

    Args:
        fig:  Matplotlib figure object.
        path: Output file path (PNG).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Shared plot primitives
# ---------------------------------------------------------------------------

def _plot_curves(
    ax_loss: plt.Axes,
    ax_acc: plt.Axes,
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    label: str,
    color: str,
) -> None:
    """Plot loss and accuracy curves onto existing axes.

    Args:
        ax_loss:    Axes for loss curves.
        ax_acc:     Axes for accuracy curves.
        epochs:     Epoch indices.
        train_loss: Training losses per epoch.
        val_loss:   Validation losses per epoch.
        train_acc:  Training accuracies per epoch.
        val_acc:    Validation accuracies per epoch.
        label:      Legend prefix for this run.
        color:      Line colour.
    """
    ax_loss.plot(epochs, train_loss, color=color, linestyle="-",  label=f"{label} train")
    ax_loss.plot(epochs, val_loss,   color=color, linestyle="--", label=f"{label} val")
    ax_acc.plot(epochs,  train_acc,  color=color, linestyle="-",  label=f"{label} train")
    ax_acc.plot(epochs,  val_acc,    color=color, linestyle="--", label=f"{label} val")


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def plot_parta_curves(results_dir: str, plots_dir: str) -> None:
    """Generate training-curve figure comparing resize and modify options.

    Saves ``parta_training_curves.png`` to *plots_dir*.

    Args:
        results_dir: Directory containing ``best_model_parta_*_results.json``.
        plots_dir:   Output directory for the figure.
    """
    resize = load_json(os.path.join(results_dir, "best_model_parta_resize_results.json"))
    modify = load_json(os.path.join(results_dir, "best_model_parta_modify_results.json"))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle("Part A - Transfer Learning: Training Curves", fontsize=13, fontweight="bold")

    ax_loss, ax_acc = axes
    colors = {"resize": "#1f77b4", "modify": "#d62728"}

    for data, color in zip([resize, modify], colors.values()):
        opt = data["option"]
        epochs, tr_l, vl_l, tr_a, vl_a = extract_history(data)
        best = data["best_val_acc"]
        _plot_curves(ax_loss, ax_acc, epochs, tr_l, vl_l, tr_a, vl_a,
                     label=opt, color=color)
        ax_acc.axhline(best, color=color, linestyle=":", linewidth=0.8,
                       label=f"{opt} best={best:.4f}")

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "parta_training_curves.png"))


def plot_parta_bar(results_dir: str, plots_dir: str) -> None:
    """Generate bar chart comparing test accuracy of resize vs modify.

    Saves ``parta_accuracy_comparison.png`` to *plots_dir*.

    Args:
        results_dir: Directory containing Part A result JSON files.
        plots_dir:   Output directory for the figure.
    """
    resize = load_json(os.path.join(results_dir, "best_model_parta_resize_results.json"))
    modify = load_json(os.path.join(results_dir, "best_model_parta_modify_results.json"))

    labels  = ["Resize + Freeze\n(FC only)", "Modify + Fine-tune\n(All layers)"]
    accs    = [resize["best_val_acc"], modify["best_val_acc"]]
    colors  = ["#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [a * 100 for a in accs], color=colors, width=0.45, edgecolor="black")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc*100:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Part A - Transfer Learning: Option Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "parta_accuracy_comparison.png"))


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def plot_partb_baseline_curves(results_dir: str, plots_dir: str) -> None:
    """Training curves for SimpleCNN, ResNet w/o LS, ResNet with LS.

    Saves ``partb_baseline_curves.png`` to *plots_dir*.

    Args:
        results_dir: Directory containing Part B result JSON files.
        plots_dir:   Output directory for the figure.
    """
    configs: List[Tuple[str, str, str]] = [
        ("best_model_simplecnn_results.json",    "SimpleCNN",       "#2ca02c"),
        ("best_model_resnet_results.json",       "ResNet (no LS)",  "#1f77b4"),
        ("best_model_resnet_ls0.1_results.json", "ResNet (LS=0.1)", "#d62728"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle("Part B - Baseline Models: Training Curves", fontsize=13, fontweight="bold")
    ax_loss, ax_acc = axes

    for fname, label, color in configs:
        data = load_json(os.path.join(results_dir, fname))
        epochs, tr_l, vl_l, tr_a, vl_a = extract_history(data)
        best = data["best_val_acc"]
        _plot_curves(ax_loss, ax_acc, epochs, tr_l, vl_l, tr_a, vl_a,
                     label=label, color=color)
        ax_acc.axhline(best, color=color, linestyle=":", linewidth=0.8,
                       label=f"{label} best={best:.4f}")

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "partb_baseline_curves.png"))


def plot_partb_kd_curves(results_dir: str, plots_dir: str) -> None:
    """Training curves for KD experiments (SimpleCNN KD and MobileNet KD).

    Saves ``partb_kd_curves.png`` to *plots_dir*.

    Args:
        results_dir: Directory containing Part B result JSON files.
        plots_dir:   Output directory for the figure.
    """
    configs: List[Tuple[str, str, str]] = [
        ("best_model_simplecnn_results.json",            "SimpleCNN (baseline)", "#2ca02c"),
        ("best_model_simplecnn_kd_results.json",         "SimpleCNN (KD)",       "#17becf"),
        ("best_model_mobilenet_modified_kd_results.json","MobileNet (mod. KD)",  "#ff7f0e"),
        ("best_model_resnet_ls0.1_results.json",         "ResNet (teacher)",     "#d62728"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle("Part B - Knowledge Distillation: Training Curves", fontsize=13, fontweight="bold")
    ax_loss, ax_acc = axes

    for fname, label, color in configs:
        data = load_json(os.path.join(results_dir, fname))
        epochs, tr_l, vl_l, tr_a, vl_a = extract_history(data)
        best = data["best_val_acc"]
        _plot_curves(ax_loss, ax_acc, epochs, tr_l, vl_l, tr_a, vl_a,
                     label=label, color=color)
        ax_acc.axhline(best, color=color, linestyle=":", linewidth=0.8,
                       label=f"{label} best={best:.4f}")

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "partb_kd_curves.png"))


def plot_partb_accuracy_bar(results_dir: str, plots_dir: str) -> None:
    """Bar chart comparing all Part B model accuracies.

    Saves ``partb_accuracy_comparison.png`` to *plots_dir*.

    Args:
        results_dir: Directory containing Part B result JSON files.
        plots_dir:   Output directory for the figure.
    """
    configs: List[Tuple[str, str, str]] = [
        ("best_model_simplecnn_results.json",            "SimpleCNN\n(baseline)",  "#2ca02c"),
        ("best_model_resnet_results.json",               "ResNet\n(no LS)",        "#1f77b4"),
        ("best_model_resnet_ls0.1_results.json",         "ResNet\n(LS=0.1)",       "#d62728"),
        ("best_model_simplecnn_kd_results.json",         "SimpleCNN\n(KD)",        "#17becf"),
        ("best_model_mobilenet_modified_kd_results.json","MobileNet\n(mod. KD)",   "#ff7f0e"),
    ]

    labels = [c[1] for c in configs]
    accs   = [load_json(os.path.join(results_dir, c[0]))["best_val_acc"] * 100 for c in configs]
    colors = [c[2] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", width=0.55)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(60, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Part B - Model Accuracy Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "partb_accuracy_comparison.png"))


def plot_partb_flops_scatter(plots_dir: str) -> None:
    """Scatter plot of accuracy vs. MACs for Part B models.

    Values are taken directly from logged experiment outputs.
    Saves ``partb_flops_vs_accuracy.png`` to *plots_dir*.

    Args:
        plots_dir: Output directory for the figure.
    """
    # (model_name, MACs, test_accuracy_percent, color)
    models: List[Tuple[str, int, float, str]] = [
        ("SimpleCNN\n(baseline)",   6_276_618,   74.32, "#2ca02c"),
        ("SimpleCNN\n(KD)",         6_276_618,   72.25, "#17becf"),
        ("MobileNet\n(mod. KD)",   96_160_266,   89.68, "#ff7f0e"),
        ("ResNet-18\n(teacher)",  557_224_970,   92.71, "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, macs, acc, color in models:
        ax.scatter(macs / 1e6, acc, s=160, color=color, zorder=3, edgecolors="black")
        ax.annotate(name, xy=(macs / 1e6, acc),
                    xytext=(8, -4), textcoords="offset points",
                    fontsize=8.5, color=color)

    ax.set_xlabel("MACs (millions)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Part B - Accuracy vs. Computational Cost (MACs)", fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}M" if x >= 1 else f"{x*1000:.0f}K")
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(65, 97)
    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "partb_flops_vs_accuracy.png"))


def plot_partb_flops_bar(plots_dir: str) -> None:
    """Grouped bar chart comparing MACs and accuracy for key Part B models.

    Saves ``partb_flops_bar.png`` to *plots_dir*.

    Args:
        plots_dir: Output directory for the figure.
    """
    models = ["ResNet-18\n(teacher)", "MobileNet\n(mod. KD)", "SimpleCNN\n(KD)"]
    macs_m = [557.22, 96.16, 6.28]
    accs   = [92.71,  89.68, 72.25]
    colors = ["#d62728", "#ff7f0e", "#17becf"]

    x = np.arange(len(models))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, macs_m, width, label="MACs (M)",      color=colors, alpha=0.75, edgecolor="black")
    ax2.bar(x + width / 2, accs,   width, label="Accuracy (%)",  color=colors, alpha=1.0,  edgecolor="black",
            hatch="//")

    ax1.set_ylabel("MACs (millions)", fontsize=10)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_title("Part B - FLOPs vs. Accuracy", fontweight="bold")
    ax1.set_ylim(0, 650)
    ax2.set_ylim(60, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "partb_flops_bar.png"))


def get_test_loader(data_dir: str, batch_size: int = 256, input_size: int = 32) -> DataLoader:
    """Build the CIFAR-10 test DataLoader.

    Args:
        data_dir:   Root directory for the CIFAR-10 dataset.
        batch_size: Batch size for inference.
        input_size: Input resolution (32 for native, 224 for resize option).

    Returns:
        DataLoader over the CIFAR-10 test split.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    tf_list = []
    if input_size == 224:
        tf_list.append(transforms.Resize(224))
    tf_list += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    tf = transforms.Compose(tf_list)
    ds = datasets.CIFAR10(data_dir, train=False, download=False, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and collect all predictions and ground-truth labels.

    Args:
        model:  The neural network (already loaded with weights).
        loader: Test DataLoader.
        device: Computation device.

    Returns:
        Tuple of ``(all_labels, all_preds)`` as NumPy arrays.
    """
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def plot_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    title: str,
    save_path: str,
    normalize: bool = True,
) -> None:
    """Compute and plot the confusion matrix for a model on the test set.

    Args:
        model:      Neural network with pretrained weights already loaded.
        loader:     Test DataLoader.
        device:     Computation device.
        title:      Figure title.
        save_path:  Output PNG path.
        normalize:  If ``True``, normalise rows to show recall per class.
    """
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    labels, preds = collect_predictions(model, loader, device)
    cm = confusion_matrix(labels, preds)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        cbar_label = "Recall"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.4, linecolor="lightgray",
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_parta_confusion_matrices(
    results_dir: str,
    plots_dir: str,
    data_dir: str,
    device: torch.device,
) -> None:
    """Generate confusion matrices for Part A resize and modify models.

    Loads model weights from ``results_dir`` and runs inference on the
    CIFAR-10 test set.

    Args:
        results_dir: Directory containing ``best_model_parta_*.pth`` files.
        plots_dir:   Output directory for figures.
        data_dir:    CIFAR-10 root data directory.
        device:      Computation device.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from helper import build_parta_model

    for option, input_size in [("resize", 224), ("modify", 32)]:
        pth = os.path.join(results_dir, f"best_model_parta_{option}.pth")
        if not os.path.exists(pth):
            print(f"  [skip] {pth} not found")
            continue

        # Build model with the same config used during training
        from params.model_params import PartAParams
        model_params = PartAParams(option=option, input_size=input_size,
                                   num_classes=10, pretrained=False)
        model = build_parta_model(model_params, device)
        model.load_state_dict(torch.load(pth, map_location=device))

        loader = get_test_loader(data_dir, input_size=input_size)
        plot_confusion_matrix(
            model, loader, device,
            title=f"Part A Confusion Matrix — {option} option",
            save_path=os.path.join(plots_dir, f"parta_confusion_{option}.png"),
        )


def plot_partb_confusion_matrices(
    results_dir: str,
    plots_dir: str,
    data_dir: str,
    device: torch.device,
) -> None:
    """Generate confusion matrices for all Part B models.

    Args:
        results_dir: Directory containing ``best_model_*.pth`` files.
        plots_dir:   Output directory for figures.
        data_dir:    CIFAR-10 root data directory.
        device:      Computation device.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from helper import build_model
    from params.model_params import get_simplecnn_params, get_resnet_params, get_mobilenet_params

    # Minimal dummy args for param builders
    class _Args:
        num_classes = 10
        resnet_layers = [2, 2, 2, 2]
        vgg_depth = "16"

    args = _Args()

    configs: List[Tuple[str, object, str, str]] = [
        ("best_model_simplecnn.pth",
         get_simplecnn_params(args),  "SimpleCNN (baseline)",    "partb_confusion_simplecnn.png"),
        ("best_model_resnet.pth",
         get_resnet_params(args),     "ResNet-18 (no LS)",       "partb_confusion_resnet_no_ls.png"),
        ("best_model_resnet_ls0.1.pth",
         get_resnet_params(args),     "ResNet-18 (LS=0.1)",      "partb_confusion_resnet_ls.png"),
        ("best_model_simplecnn_kd.pth",
         get_simplecnn_params(args),  "SimpleCNN (KD)",          "partb_confusion_simplecnn_kd.png"),
        ("best_model_mobilenet_modified_kd.pth",
         get_mobilenet_params(args),  "MobileNetV2 (mod. KD)",   "partb_confusion_mobilenet.png"),
    ]

    loader = get_test_loader(data_dir, input_size=32)

    for fname, model_params, title, out_name in configs:
        pth = os.path.join(results_dir, fname)
        if not os.path.exists(pth):
            print(f"  [skip] {pth} not found")
            continue
        model = build_model(model_params).to(device)
        model.load_state_dict(torch.load(pth, map_location=device))
        plot_confusion_matrix(
            model, loader, device,
            title=f"Part B Confusion Matrix — {title}",
            save_path=os.path.join(plots_dir, out_name),
        )


def plot_label_smoothing_effect(results_dir: str, plots_dir: str) -> None:
    """Validation accuracy curves highlighting the label smoothing effect on ResNet.

    Saves ``partb_label_smoothing.png`` to *plots_dir*.

    Args:
        results_dir: Directory containing Part B result JSON files.
        plots_dir:   Output directory for the figure.
    """
    no_ls = load_json(os.path.join(results_dir, "best_model_resnet_results.json"))
    ls    = load_json(os.path.join(results_dir, "best_model_resnet_ls0.1_results.json"))

    fig, ax = plt.subplots(figsize=(7, 4))

    for data, label, color, ls_style in [
        (no_ls, "ResNet (no LS)",  "#1f77b4", "-"),
        (ls,    "ResNet (LS=0.1)", "#d62728", "--"),
    ]:
        epochs, _, _, _, val_acc = extract_history(data)
        best = data["best_val_acc"]
        ax.plot(epochs, val_acc, color=color, linestyle=ls_style,
                linewidth=1.8, label=f"{label}  (best={best:.4f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Part B - Effect of Label Smoothing on ResNet-18", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "partb_label_smoothing.png"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Generate plots for CS515 HW2")
    parser.add_argument(
        "--hw_part",
        choices=["PART_A", "PART_B", "ALL"],
        default="ALL",
        help="Which part to visualise (default: ALL)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root results directory (default: results/)",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Root output directory for figures (default: plots/)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="CIFAR-10 root directory for confusion matrix inference (default: ./data)",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        default=True,
        help="Generate confusion matrices (default: True). Pass --no_confusion to skip.",
    )
    parser.add_argument(
        "--no_confusion",
        dest="confusion",
        action="store_false",
        help="Skip confusion matrix generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for confusion matrix inference (default: cuda if available)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: generate all requested figures."""
    args = get_args()

    parta_results = os.path.join(args.results_dir, "parta")
    partb_results = os.path.join(args.results_dir, "partb")
    parta_plots   = os.path.join(args.plots_dir,   "parta")
    partb_plots   = os.path.join(args.plots_dir,   "partb")

    device = torch.device(args.device)

    if args.hw_part in ("PART_A", "ALL"):
        print("\n=== Part A Plots ===")
        plot_parta_curves(parta_results, parta_plots)
        plot_parta_bar(parta_results, parta_plots)
        if args.confusion:
            print("\n--- Part A Confusion Matrices ---")
            plot_parta_confusion_matrices(parta_results, parta_plots, args.data_dir, device)

    if args.hw_part in ("PART_B", "ALL"):
        print("\n=== Part B Plots ===")
        plot_partb_baseline_curves(partb_results, partb_plots)
        plot_partb_kd_curves(partb_results, partb_plots)
        plot_partb_accuracy_bar(partb_results, partb_plots)
        plot_label_smoothing_effect(partb_results, partb_plots)
        plot_partb_flops_scatter(partb_plots)
        plot_partb_flops_bar(partb_plots)
        if args.confusion:
            print("\n--- Part B Confusion Matrices ---")
            plot_partb_confusion_matrices(partb_results, partb_plots, args.data_dir, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
