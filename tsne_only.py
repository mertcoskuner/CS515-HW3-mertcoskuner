"""Standalone t-SNE + combined GradCAM visualisation script for PART_C.

Loads already-trained PART_A/modify original and AugMix checkpoints,
runs PGD-20 L∞ on the full test set, and produces:
  - t-SNE plots (n_samples per model, individual)
  - Combined GradCAM figure (both models side-by-side, max_samples rows)
"""

import argparse
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights

from attacks import pgd_attack
from gradcam import (
    GradCAM,
    CIFAR10_CLASSES,
    denormalize,
    get_target_layer,
    overlay_heatmap,
    visualize_tsne,
)
from helper import build_parta_model, set_seed
from params.model_params import PartAParams


def _load_model(ckpt: str, device: torch.device) -> torch.nn.Module:
    params = PartAParams(
        model="PartA",
        num_classes=10,
        pretrained=False,
        weights=None,
        option="modify",
        input_size=32,
    )
    m = build_parta_model(params, device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def _collect_adv(model, loader, device, eps: float = 4 / 255, steps: int = 20):
    """Run PGD on the full loader and return (clean, adv, labels) tensors."""
    all_clean, all_adv, all_labels = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv = pgd_attack(model, imgs, labels, norm="linf",
                         eps=eps, alpha=eps / 4, num_steps=steps)
        all_clean.append(imgs.cpu())
        all_adv.append(adv.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_clean), torch.cat(all_adv), torch.cat(all_labels)


def visualize_gradcam_combined(
    models_dict: "OrderedDict[str, torch.nn.Module]",
    clean_imgs: torch.Tensor,
    adv_linf_imgs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    save_path: str = "gradcam_combined.png",
    max_samples: int = 4,
) -> None:
    """Plot GradCAM side-by-side for multiple models on the same samples.

    Selects up to *max_samples* images that are misclassified under L∞ PGD
    for the first model, then shows all models on those same samples.
    Layout: rows = samples, column groups = models (4 cols each:
    Clean | GradCAM(clean) | Adversarial | GradCAM(adv)).

    Args:
        models_dict: OrderedDict of {name: model} shown left to right.
        clean_imgs:    Normalised clean images [B, C, H, W].
        adv_linf_imgs: Matching L∞ adversarial images [B, C, H, W].
        labels:        True class labels [B].
        device:        Computation device.
        save_path:     Output PNG path.
        max_samples:   Maximum rows to visualise.
    """
    model_list  = list(models_dict.items())
    ref_model   = model_list[0][1]

    with torch.no_grad():
        ref_clean_preds = ref_model(clean_imgs.to(device)).argmax(1).cpu()
        ref_adv_preds   = ref_model(adv_linf_imgs.to(device)).argmax(1).cpu()

    labels_cpu = labels.cpu()
    selected = [
        i for i in range(len(labels_cpu))
        if ref_clean_preds[i] == labels_cpu[i] and ref_adv_preds[i] != labels_cpu[i]
    ][:max_samples]

    if not selected:
        print("  [GradCAM combined] No misclassified samples found.")
        return

    n_models      = len(model_list)
    cols_per_model = 4  # Clean | GradCAM(clean) | Adversarial | GradCAM(adv)
    total_cols    = n_models * cols_per_model

    fig, axes = plt.subplots(
        len(selected), total_cols,
        figsize=(cols_per_model * n_models * 3, 3.5 * len(selected)),
    )
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean", "GradCAM\n(clean)", "Adversarial", "GradCAM\n(adv)"]
    for m_idx, (m_name, _) in enumerate(model_list):
        base = m_idx * cols_per_model
        axes[0, base].set_title(f"{m_name}\n{col_titles[0]}", fontsize=8, fontweight="bold")
        for c, t in enumerate(col_titles[1:], 1):
            axes[0, base + c].set_title(t, fontsize=8)

    for row, idx in enumerate(selected):
        true_lbl = CIFAR10_CLASSES[labels_cpu[idx]]
        x_clean  = clean_imgs[idx].unsqueeze(0).to(device)
        x_adv    = adv_linf_imgs[idx].unsqueeze(0).to(device)
        img_c    = denormalize(x_clean.cpu())
        img_a    = denormalize(x_adv.cpu())

        axes[row, 0].set_ylabel(f"True: {true_lbl}", fontsize=8,
                                rotation=0, labelpad=60, va="center")

        for m_idx, (m_name, model) in enumerate(model_list):
            model.eval()
            target_layer = get_target_layer(model)
            base = m_idx * cols_per_model

            with torch.no_grad():
                c_pred = model(x_clean).argmax(1).item()
                a_pred = model(x_adv).argmax(1).item()

            gcam_c = GradCAM(model, target_layer)
            heat_c, _ = gcam_c(x_clean.clone().requires_grad_(True),
                               class_idx=labels_cpu[idx].item())
            gcam_c.remove_hooks()

            gcam_a = GradCAM(model, target_layer)
            heat_a, _ = gcam_a(x_adv.clone().requires_grad_(True), class_idx=a_pred)
            gcam_a.remove_hooks()

            axes[row, base + 0].imshow(img_c)
            axes[row, base + 0].set_xlabel(f"→ {CIFAR10_CLASSES[c_pred]}", fontsize=7)
            axes[row, base + 1].imshow(overlay_heatmap(img_c, heat_c))
            axes[row, base + 2].imshow(img_a)
            axes[row, base + 2].set_xlabel(f"→ {CIFAR10_CLASSES[a_pred]}", fontsize=7)
            axes[row, base + 3].imshow(overlay_heatmap(img_a, heat_a))

            for c in range(cols_per_model):
                axes[row, base + c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [GradCAM combined] Saved → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     type=str, default="./data")
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--num_workers",  type=int, default=8)
    parser.add_argument("--n_samples",    type=int, default=2000)
    parser.add_argument("--max_samples",  type=int, default=4,
                        help="Max GradCAM rows per figure.")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vanilla_ckpt", type=str,
                        default="results/parta/best_model_parta_modify.pth")
    parser.add_argument("--augmix_ckpt",  type=str,
                        default="results/partc/best_model_parta_modify_augmix.pth")
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    checkpoints = OrderedDict([
        ("Vanilla",  args.vanilla_ckpt),
        ("AugMix",   args.augmix_ckpt),
    ])

    # Check all checkpoints exist
    for name, ckpt in checkpoints.items():
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"{name} checkpoint not found: {ckpt}")

    os.makedirs("plots/partc", exist_ok=True)

    # Load models and collect adversarial examples (PGD on full test set per model)
    loaded_models = OrderedDict()
    adv_data = {}

    for name, ckpt in checkpoints.items():
        print(f"\n[{name}] Loading checkpoint: {ckpt}")
        model = _load_model(ckpt, device)
        loaded_models[name] = model

        print(f"[{name}] Running PGD-20 on full test set ...")
        clean, adv, labels = _collect_adv(model, loader, device)
        adv_data[name] = (clean, adv, labels)

        # Individual t-SNE
        visualize_tsne(
            model, clean, adv, labels, device,
            save_path=f"plots/partc/tsne_{name}_linf_full.png",
            n_samples=args.n_samples,
        )

    # Combined GradCAM — use vanilla model's adversarial examples for sample selection
    print("\n[GradCAM combined] Generating side-by-side figure ...")
    clean_ref, adv_ref, labels_ref = adv_data["Vanilla"]
    visualize_gradcam_combined(
        loaded_models, clean_ref, adv_ref, labels_ref, device,
        save_path="plots/partc/gradcam_combined_linf.png",
        max_samples=args.max_samples,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
