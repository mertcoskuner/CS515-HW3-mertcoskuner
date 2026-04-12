import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from attacks import pgd_attack
from params.data_params import Cifar10Params
from params.model_training_params import ModelTrainingConfig

CIFAR10C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]


class _Cifar10CDataset(Dataset):
    """Minimal Dataset wrapper for a numpy image/label slice."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self.labels[idx])


@torch.no_grad()
def run_test(
    model: nn.Module,
    data_params: Cifar10Params,
    model_training_params: ModelTrainingConfig,
    device: torch.device,
    model_name: str = "",
    input_size: int = 32,
) -> float:
    """Evaluate a model on the CIFAR-10 test set and print per-class accuracy.

    Args:
        model: Model to evaluate (weights must already be loaded).
        data_params: Cifar10Params with normalisation statistics.
        model_training_params: ModelTrainingConfig with data_dir, batch_size, num_workers.
        device: Device to run inference on.
        model_name: Optional label shown in the printed header.
        input_size: Spatial size of inputs. Use 224 for PART_A resize option;
            32 for all other models.

    Returns:
        Overall test accuracy as a float in [0, 1].
    """
    if input_size == 224:
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])

    val_ds = datasets.CIFAR10(
        model_training_params.data_dir, train=False, download=True, transform=transform_test,
    )
    loader = DataLoader(
        val_ds,
        batch_size=model_training_params.device_config.batch_size,
        shuffle=False,
        num_workers=model_training_params.num_workers,
    )

    model.eval()
    correct, n = 0, 0
    class_correct = [0] * 10
    class_total   = [0] * 10

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    tag = f" [{model_name}]" if model_name else ""
    print(f"\n=== Test Results{tag} ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(10):
        acc = class_correct[i] / class_total[i]
        print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    return correct / n


@torch.no_grad()
def run_cifar10c_test(
    model: nn.Module,
    data_params: Cifar10Params,
    cifar10c_dir: str,
    model_training_params: ModelTrainingConfig,
    device: torch.device,
    model_name: str = "",
    input_size: int = 32,
) -> dict:
    """Evaluate a model on every corruption and severity level of CIFAR-10-C.

    The CIFAR-10-C directory must contain one .npy file per corruption type
    (shape [50000, 32, 32, 3], uint8) and a labels.npy file (shape [50000]).
    The 50 000 images are laid out as five consecutive blocks of 10 000, one
    block per severity level (1 → 5).

    Args:
        model: Model to evaluate (weights must already be loaded).
        data_params: Cifar10Params with normalisation statistics.
        cifar10c_dir: Path to the directory containing CIFAR-10-C .npy files.
        model_training_params: ModelTrainingConfig with batch_size and num_workers.
        device: Device to run inference on.
        model_name: Optional label shown in the printed header.
        input_size: Spatial size of inputs. Use 224 for PART_A resize option;
            32 for all other models.

    Returns:
        Nested dict: results[corruption][severity] = accuracy (float in [0, 1]).
        Also contains results["mean_per_severity"][severity] and results["overall"].
    """
    if input_size == 224:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])

    labels_all = np.load(os.path.join(cifar10c_dir, "labels.npy"))  # (50000,)

    results: dict = {}
    model.eval()

    for corruption in CIFAR10C_CORRUPTIONS:
        npy_path = os.path.join(cifar10c_dir, f"{corruption}.npy")
        if not os.path.isfile(npy_path):
            print(f"  [WARNING] {npy_path} not found — skipping.")
            continue

        images_all = np.load(npy_path)  # (50000, 32, 32, 3)
        results[corruption] = {}

        for sev in range(1, 6):
            start = (sev - 1) * 10_000
            end = sev * 10_000
            images = images_all[start:end]
            labels = labels_all[start:end]

            ds = _Cifar10CDataset(images, labels, transform)
            loader = DataLoader(
                ds,
                batch_size=model_training_params.device_config.batch_size,
                shuffle=False,
                num_workers=model_training_params.num_workers,
            )

            correct, n = 0, 0
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs).argmax(1)
                correct += preds.eq(lbls).sum().item()
                n += imgs.size(0)

            results[corruption][sev] = correct / n

    # Aggregate across corruptions per severity and overall
    valid = [c for c in CIFAR10C_CORRUPTIONS if c in results]
    mean_per_sev = {}
    for sev in range(1, 6):
        accs = [results[c][sev] for c in valid if sev in results[c]]
        mean_per_sev[sev] = sum(accs) / len(accs) if accs else float("nan")

    all_accs = [results[c][s] for c in valid for s in range(1, 6) if s in results[c]]
    overall = sum(all_accs) / len(all_accs) if all_accs else float("nan")

    results["mean_per_severity"] = mean_per_sev
    results["overall"] = overall

    # ── Print results table ───────────────────────────────────────────────────
    tag = f" [{model_name}]" if model_name else ""
    col_w = 18
    print(f"\n=== CIFAR-10-C Results{tag} ===")
    header = f"  {'Corruption':<25}" + "".join(f"  Sev{s:>1}" for s in range(1, 6))
    print(header)
    print("-" * (25 + 10 * 5 + 4))
    for corruption in valid:
        row = f"  {corruption:<25}"
        for sev in range(1, 6):
            row += f"  {results[corruption][sev]:.3f}"
        print(row)
    print("-" * (25 + 10 * 5 + 4))
    mean_row = f"  {'Mean':<25}"
    for sev in range(1, 6):
        mean_row += f"  {mean_per_sev[sev]:.3f}"
    print(mean_row)
    print(f"\n  Overall mean accuracy: {overall:.4f}")

    return results


def run_pgd_test(
    model: nn.Module,
    data_params: Cifar10Params,
    model_training_params: ModelTrainingConfig,
    device: torch.device,
    model_name: str = "",
    input_size: int = 32,
    eps_linf: float = 4 / 255,
    eps_l2: float = 0.25,
    num_steps: int = 20,
    collect_samples: bool = False,
) -> dict:
    """Evaluate adversarial robustness under PGD20 with L∞ and L2 constraints.

    For each test batch, generates adversarial examples with both norms and
    measures accuracy on clean, L∞-adversarial, and L2-adversarial inputs.

    Args:
        model:                Model to evaluate (weights must already be loaded).
        data_params:          Cifar10Params with normalisation statistics.
        model_training_params: ModelTrainingConfig with data_dir, batch_size, num_workers.
        device:               Device to run inference on.
        model_name:           Optional label shown in the printed header.
        input_size:           Spatial size (32 or 224 for PART_A resize).
        eps_linf:             L∞ perturbation budget in pixel space (default 4/255).
        eps_l2:               L2 perturbation radius in pixel space (default 0.25).
        num_steps:            Number of PGD steps (default 20).
        collect_samples:      If True, also return one batch of (clean, adv_linf, adv_l2, labels)
                              for GradCAM and t-SNE visualisation.

    Returns:
        Dict with keys 'clean_acc', 'linf_acc', 'l2_acc', and optionally
        'samples' → (clean_imgs, adv_linf_imgs, adv_l2_imgs, labels).
    """
    if input_size == 224:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])

    ds = datasets.CIFAR10(
        model_training_params.data_dir, train=False, download=True, transform=transform,
    )
    loader = DataLoader(
        ds,
        batch_size=model_training_params.device_config.batch_size,
        shuffle=False,
        num_workers=model_training_params.num_workers,
    )

    model.eval()
    clean_correct, linf_correct, l2_correct, n = 0, 0, 0, 0
    sample_batch = None

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            clean_preds = model(imgs).argmax(1)
            clean_correct += clean_preds.eq(labels).sum().item()

        adv_linf = pgd_attack(model, imgs, labels, norm="linf",
                              eps=eps_linf, alpha=eps_linf / 4, num_steps=num_steps)
        adv_l2   = pgd_attack(model, imgs, labels, norm="l2",
                              eps=eps_l2,   alpha=eps_l2 / 5,   num_steps=num_steps)

        with torch.no_grad():
            linf_correct += model(adv_linf).argmax(1).eq(labels).sum().item()
            l2_correct   += model(adv_l2).argmax(1).eq(labels).sum().item()

        n += imgs.size(0)

        if collect_samples and sample_batch is None:
            sample_batch = (
                imgs.cpu(), adv_linf.cpu(), adv_l2.cpu(), labels.cpu(),
            )

    tag = f" [{model_name}]" if model_name else ""
    print(f"\n=== PGD-{num_steps} Adversarial Robustness{tag} ===")
    print(f"  Clean accuracy :      {clean_correct/n:.4f}  ({clean_correct}/{n})")
    print(f"  L∞  (ε={eps_linf:.4f}) : {linf_correct/n:.4f}  ({linf_correct}/{n})")
    print(f"  L2  (ε={eps_l2:.4f})   : {l2_correct/n:.4f}  ({l2_correct}/{n})")

    result = {
        "clean_acc": clean_correct / n,
        "linf_acc":  linf_correct  / n,
        "l2_acc":    l2_correct    / n,
    }
    if collect_samples:
        result["samples"] = sample_batch
    return result


def run_transfer_test(
    source_model: nn.Module,
    target_model: nn.Module,
    data_params: Cifar10Params,
    model_training_params: ModelTrainingConfig,
    device: torch.device,
    source_name: str = "source",
    target_name: str = "target",
    eps_linf: float = 4 / 255,
    num_steps: int = 20,
) -> dict:
    """Evaluate adversarial transferability of PGD examples from source to target model.

    Generates adversarial examples using PGD-20 L∞ on *source_model* (teacher),
    then measures accuracy of both source and target models on those examples.
    This reveals how well white-box adversarial examples transfer to the student.

    Args:
        source_model: Model used to generate adversarial examples (teacher).
        target_model: Model tested on source-generated adversarial examples (student).
        data_params:  Cifar10Params with normalisation statistics.
        model_training_params: ModelTrainingConfig with data_dir, batch_size, num_workers.
        device:       Computation device.
        source_name:  Label for the source model in printed output.
        target_name:  Label for the target model in printed output.
        eps_linf:     L∞ perturbation budget (default 4/255).
        num_steps:    PGD iterations (default 20).

    Returns:
        Dict with keys 'source_clean_acc', 'source_adv_acc',
        'target_clean_acc', 'target_adv_acc'.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_params.mean, std=data_params.std),
    ])
    ds = datasets.CIFAR10(
        model_training_params.data_dir, train=False, download=True, transform=transform,
    )
    loader = DataLoader(
        ds,
        batch_size=model_training_params.device_config.batch_size,
        shuffle=False,
        num_workers=model_training_params.num_workers,
    )

    source_model.eval()
    target_model.eval()
    src_clean, src_adv, tgt_clean, tgt_adv, n = 0, 0, 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        adv_imgs = pgd_attack(source_model, imgs, labels, norm="linf",
                              eps=eps_linf, alpha=eps_linf / 4, num_steps=num_steps)

        with torch.no_grad():
            src_clean += source_model(imgs).argmax(1).eq(labels).sum().item()
            src_adv   += source_model(adv_imgs).argmax(1).eq(labels).sum().item()
            tgt_clean += target_model(imgs).argmax(1).eq(labels).sum().item()
            tgt_adv   += target_model(adv_imgs).argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

    print(f"\n=== Transferability: {source_name} → {target_name} (PGD-{num_steps} L∞ ε={eps_linf:.4f}) ===")
    print(f"  {source_name:<30}  clean: {src_clean/n:.4f}  adv: {src_adv/n:.4f}")
    print(f"  {target_name:<30}  clean: {tgt_clean/n:.4f}  adv: {tgt_adv/n:.4f}")

    return {
        "source_clean_acc": src_clean / n,
        "source_adv_acc":   src_adv   / n,
        "target_clean_acc": tgt_clean / n,
        "target_adv_acc":   tgt_adv   / n,
    }
