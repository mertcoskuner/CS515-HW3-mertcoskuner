"""
GradCAM + t-SNE visualisation utilities for CIFAR-10 adversarial analysis.

GradCAM reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization," ICCV 2017. https://arxiv.org/abs/1610.02391

Key idea
--------
After a forward pass we:
  1. Pick the last convolutional layer (richest spatial features).
  2. Compute  dY^c / dA^k  — gradient of class score w.r.t. each activation map.
  3. Global-average-pool gradients → importance weights  α_k^c.
  4. Weighted sum of activation maps, then ReLU:

     GradCAM(x) = ReLU( Σ_k  α_k^c · A^k )

  5. Resize heatmap to input resolution and overlay on the image.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE

CIFAR10_MEAN   = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD    = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── GradCAM core ───────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping (model-agnostic).

    How the hooks work
    ------------------
    PyTorch's hook system lets us intercept:
      • forward_hook       — captures activations A^k after the forward pass.
      • full_backward_hook — captures gradients dY/dA^k after the backward pass.

    We register both on the target conv layer, then read them off after
    loss.backward() to compute the weighted heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._fwd_hook   = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        self._bwd_hook   = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """Compute GradCAM heatmap for input *x* and class *class_idx*.

        Args:
            x:         Single image tensor [1, C, H, W] (normalised).
            class_idx: Target class index.  If None, uses the predicted class.

        Returns:
            Tuple of (heatmap, class_idx) where heatmap is a float ndarray
            of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[0, class_idx].backward()

        # α_k^c = global-average-pool of gradients
        weights  = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, K, 1, 1)
        cam      = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, H', W')

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        h, w    = x.shape[2], x.shape[3]
        heatmap = np.array(
            Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
        ) / 255.0
        return heatmap, class_idx

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Architecture helpers ───────────────────────────────────────────────────────

def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional layer for GradCAM for each architecture.

    Supported models:
      - ResNet (custom, from models/ResNet.py)      → layer4[-1]
      - SimpleCNN                                   → conv2
      - MobileNetV2                                 → conv2  (1280-ch pointwise)
      - torchvision ResNet-18  (PART_A)             → layer4[-1]
    """
    name = type(model).__name__
    if name == "ResNet":
        return model.layer4[-1]
    if name == "SimpleCNN":
        return model.conv2
    if name == "MobileNetV2":
        return model.conv2          # shape (B, 1280, H, W) — last feature conv
    if hasattr(model, "layer4"):    # torchvision ResNet-18
        return model.layer4[-1]
    raise ValueError(f"Cannot determine GradCAM target layer for '{name}'")


def get_feature_layer(model: nn.Module) -> nn.Module:
    """Return the layer whose output is used as features for t-SNE.

    The hook on this layer captures the penultimate representation before
    the final classifier head.

    Supported models:
      - ResNet (custom)    → avgpool    output (B, 512, 1, 1) → will be flattened
      - SimpleCNN          → fc1        output (B, 128)
      - MobileNetV2        → conv2      output (B, 1280, H, W) → will be pooled+flattened
      - torchvision ResNet → avgpool    output (B, 512, 1, 1) → will be flattened
    """
    name = type(model).__name__
    if name == "ResNet":
        return model.avgpool
    if name == "SimpleCNN":
        return model.fc1
    if name == "MobileNetV2":
        return model.conv2
    if hasattr(model, "avgpool"):
        return model.avgpool
    raise ValueError(f"Cannot determine feature layer for '{name}'")


# ── Image utilities ────────────────────────────────────────────────────────────

def denormalize(t: torch.Tensor) -> np.ndarray:
    """CIFAR-10 denormalisation → HWC numpy array in [0, 1]."""
    img = t.clone().squeeze(0) * CIFAR10_STD.squeeze(0) + CIFAR10_MEAN.squeeze(0)
    return img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a jet-coloured heatmap with the original image."""
    rgb = cm.get_cmap("jet")(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * rgb, 0, 1)


# ── GradCAM adversarial visualisation ─────────────────────────────────────────

def visualize_gradcam_adversarial(
    model: nn.Module,
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    save_path: str = "gradcam_adversarial.png",
    max_samples: int = 2,
) -> None:
    """Plot GradCAM for clean and adversarial samples that cause misclassification.

    Selects up to *max_samples* images where:
      - The model predicts the correct class for the **clean** image.
      - The model predicts the **wrong** class for the adversarial image.

    For each selected sample, five columns are shown:
      Clean | GradCAM (clean) | Adversarial | GradCAM (adv) | Perturbation ×10

    Args:
        model:       Model to probe (weights must already be loaded).
        clean_imgs:  Batch of normalised clean images [B, C, H, W].
        adv_imgs:    Matching adversarial images [B, C, H, W].
        labels:      True class labels [B].
        device:      Computation device.
        save_path:   Output PNG path.
        max_samples: Maximum number of samples to visualise.
    """
    model.eval()
    target_layer = get_target_layer(model)

    with torch.no_grad():
        clean_preds = model(clean_imgs.to(device)).argmax(1).cpu()
        adv_preds   = model(adv_imgs.to(device)).argmax(1).cpu()

    labels_cpu = labels.cpu()
    selected   = [
        i for i in range(len(labels_cpu))
        if clean_preds[i] == labels_cpu[i] and adv_preds[i] != labels_cpu[i]
    ][:max_samples]

    if not selected:
        print("  [GradCAM] No misclassified adversarial samples found in this batch.")
        return

    fig, axes = plt.subplots(len(selected), 5, figsize=(15, 3.5 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(
        ["Clean", "GradCAM (clean)", "Adversarial", "GradCAM (adv)", "Perturbation ×10"]
    ):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, idx in enumerate(selected):
        true_lbl  = CIFAR10_CLASSES[labels_cpu[idx]]
        clean_lbl = CIFAR10_CLASSES[clean_preds[idx]]
        adv_lbl   = CIFAR10_CLASSES[adv_preds[idx]]

        x_clean = clean_imgs[idx].unsqueeze(0).to(device)
        x_adv   = adv_imgs[idx].unsqueeze(0).to(device)

        # GradCAM on clean — backprop through the true class
        gcam_c = GradCAM(model, target_layer)
        heat_c, _ = gcam_c(x_clean.clone().requires_grad_(True), class_idx=labels_cpu[idx].item())
        gcam_c.remove_hooks()

        # GradCAM on adversarial — backprop through the (wrong) predicted class
        gcam_a = GradCAM(model, target_layer)
        heat_a, _ = gcam_a(x_adv.clone().requires_grad_(True), class_idx=adv_preds[idx].item())
        gcam_a.remove_hooks()

        img_c = denormalize(x_clean.cpu())
        img_a = denormalize(x_adv.cpu())
        perturb = np.clip(np.abs(img_a - img_c) * 10, 0, 1)

        axes[row, 0].imshow(img_c)
        axes[row, 0].set_ylabel(
            f"True: {true_lbl}\nClean → {clean_lbl}", fontsize=8,
            rotation=0, labelpad=90, va="center",
        )
        axes[row, 1].imshow(overlay_heatmap(img_c, heat_c))
        axes[row, 2].imshow(img_a)
        axes[row, 2].set_ylabel(
            f"Adv → {adv_lbl}", fontsize=8,
            rotation=0, labelpad=60, va="center",
        )
        axes[row, 3].imshow(overlay_heatmap(img_a, heat_a))
        axes[row, 4].imshow(perturb)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [GradCAM] Saved → {save_path}")


# ── t-SNE visualisation ────────────────────────────────────────────────────────

def _extract_features(
    model: nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract penultimate-layer features for a batch of images.

    Hooks the layer returned by ``get_feature_layer(model)``, pools spatial
    dimensions if needed, and returns a 2-D float32 array of shape (N, D).
    """
    feat_layer = get_feature_layer(model)
    feats_list = []
    captured   = [None]

    def _hook(m, i, o):
        captured[0] = o.detach()

    handle = feat_layer.register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(imgs), batch_size):
            batch = imgs[start: start + batch_size].to(device)
            model(batch)
            f = captured[0]
            if f.dim() > 2:
                f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            feats_list.append(f.cpu().numpy())

    handle.remove()
    return np.concatenate(feats_list, axis=0)


def visualize_tsne(
    model: nn.Module,
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    save_path: str = "tsne_adversarial.png",
    n_samples: int = 500,
    perplexity: int = 30,
) -> None:
    """t-SNE plot showing clean and adversarial samples in feature space.

    Uses up to *n_samples* images.  Points are coloured by ground-truth class;
    clean samples are shown as filled circles, adversarial as ×-markers.

    Args:
        model:      Model whose penultimate features are used.
        clean_imgs: Normalised clean images [N, C, H, W].
        adv_imgs:   Matching adversarial images [N, C, H, W].
        labels:     True class labels [N].
        device:     Computation device.
        save_path:  Output PNG path.
        n_samples:  Maximum number of image pairs to embed (default 500).
        perplexity: t-SNE perplexity parameter (default 30).
    """
    n = min(n_samples, len(labels))
    clean_sub = clean_imgs[:n]
    adv_sub   = adv_imgs[:n]
    labels_np = labels[:n].cpu().numpy()

    print(f"  [t-SNE] Extracting features for {n} clean + {n} adversarial samples ...")
    feats_clean = _extract_features(model, clean_sub, device)
    feats_adv   = _extract_features(model, adv_sub,   device)

    all_feats  = np.concatenate([feats_clean, feats_adv], axis=0)
    all_labels = np.concatenate([labels_np,   labels_np],  axis=0)
    source     = np.array(["clean"] * n + ["adversarial"] * n)

    print("  [t-SNE] Running TSNE ...")
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(all_feats)

    palette = plt.cm.get_cmap("tab10", 10)
    fig, ax  = plt.subplots(figsize=(9, 7))

    for cls in range(10):
        mask_clean = (all_labels == cls) & (source == "clean")
        mask_adv   = (all_labels == cls) & (source == "adversarial")
        color      = palette(cls)
        label_name = CIFAR10_CLASSES[cls]

        ax.scatter(emb[mask_clean, 0], emb[mask_clean, 1],
                   c=[color], marker="o", s=18, alpha=0.7,
                   label=f"{label_name} (clean)")
        ax.scatter(emb[mask_adv, 0],   emb[mask_adv, 1],
                   c=[color], marker="x", s=22, alpha=0.9,
                   label=f"{label_name} (adv)")

    # Custom legend: one entry per class (colour) + shape legend
    handles, lbls = ax.get_legend_handles_labels()
    # Show only the 10 class colours (clean entries) + shape guide
    ax.legend(handles[:10], [CIFAR10_CLASSES[i] for i in range(10)],
              loc="upper right", fontsize=7, ncol=2, title="Class (●=clean  ×=adv)")

    ax.set_title("t-SNE: Clean vs. Adversarial Sample Embeddings", fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [t-SNE] Saved → {save_path}")
