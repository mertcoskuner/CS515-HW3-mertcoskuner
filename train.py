import copy
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any, Dict, Tuple

from augmix import AugMixDataset, jsd_loss
from params.model_params import PartAParams
from params.model_training_params import ModelTrainingConfig
from params.data_params import Cifar10Params


def get_transforms(params: Dict[str, Any], train: bool = True) -> transforms.Compose:
    """Build a torchvision transform pipeline for CIFAR-10 or MNIST.

    Args:
        params: Dictionary containing 'mean' and 'std' normalisation values.
        train: If True, applies random augmentation; otherwise only normalises.

    Returns:
        Composed transform pipeline.
    """
    mean, std = params["mean"], params["std"]

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_loaders(params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a params dictionary.

    Args:
        params: Dictionary with keys 'dataset', 'data_dir', 'batch_size',
            'num_workers', 'mean', and 'std'.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True,  num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, num_workers=params["num_workers"])
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    """Run a single training epoch over the given DataLoader.

    Args:
        model: The model to train.
        loader: DataLoader for the training set.
        optimizer: Optimiser used to update model parameters.
        criterion: Loss function.
        device: Device to run computation on.
        log_interval: Number of batches between progress log prints.

    Returns:
        Tuple of (average_loss, accuracy) computed over the full epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def train_one_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    temperature: float,
    alpha: float,
    log_interval: int,
) -> Tuple[float, float]:
    """Run one standard KD training epoch (Hinton et al., 2015).

    Loss = alpha * T² * KL(softmax(s/T) || softmax(t/T))
         + (1 - alpha) * CE(s, y_hard)

    Args:
        student: Student network being trained.
        teacher: Frozen teacher network providing soft targets.
        loader: Training DataLoader.
        optimizer: Optimiser instance.
        criterion: Hard-label loss (CrossEntropyLoss).
        device: Computation device.
        temperature: Distillation temperature T (> 1 softens distributions).
        alpha: Weight on soft KD loss; (1-alpha) weights hard CE loss.
        log_interval: Batches between progress prints.

    Returns:
        Tuple of (mean_loss, accuracy) over the epoch.
    """
    student.train()
    teacher.eval()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(imgs)

        student_logits = student(imgs)

        ce_loss = criterion(student_logits, labels)

        soft_labels      = F.softmax(teacher_logits / temperature, dim=1)
        log_soft_student = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(log_soft_student, soft_labels, reduction="batchmean") * (temperature ** 2)

        loss = alpha * kd_loss + (1.0 - alpha) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += student_logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def train_one_epoch_teacher_prob(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    """Run one training epoch using teacher-probability dynamic label smoothing.

    The teacher's raw softmax confidence on the true class becomes a per-sample
    smoothing weight (T=1, no temperature scaling):
        - true class  → p_teacher(y_true)
        - other classes → (1 - p_teacher(y_true)) / (C - 1)  equally

    Loss = CE(student_logits, soft_target)

    High teacher confidence → near one-hot target (easy sample).
    Low teacher confidence  → spread distribution (hard sample).

    Args:
        student: Student network being trained.
        teacher: Frozen teacher network providing per-sample confidence.
        loader: Training DataLoader.
        optimizer: Optimiser instance.
        device: Computation device.
        log_interval: Batches between progress prints.

    Returns:
        Tuple of (mean_loss, accuracy) over the epoch.
    """
    student.train()
    teacher.eval()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            teacher_probs = F.softmax(teacher(imgs), dim=1)                      # [B, C] at T=1
            true_prob     = teacher_probs.gather(1, labels.unsqueeze(1))         # [B, 1]

        num_classes = teacher_probs.size(1)
        other_prob  = (1.0 - true_prob) / (num_classes - 1)                     # [B, 1]
        soft_labels = other_prob.expand_as(teacher_probs).clone()
        soft_labels.scatter_(1, labels.unsqueeze(1), true_prob)

        student_logits = student(imgs)
        loss = -(soft_labels * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += student_logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Args:
        model: Model to evaluate.
        loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run computation on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def run_training(
    model: nn.Module,
    model_params: Any,
    model_training_params: ModelTrainingConfig,
    data_params: Cifar10Params,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Full training loop for PART_B models trained from scratch.

    Args:
        model: Model to train.
        model_params: Model-specific params dataclass; model_params.model is used
            to derive the checkpoint filename.
        model_training_params: ModelTrainingConfig with all training hyperparameters.
            model_training_params.label_smoothing controls CrossEntropyLoss smoothing;
            0.0 disables it.
        data_params: Cifar10Params with normalisation statistics.

    Returns:
        Tuple of (best_model, results) where results is a dict containing 'model',
        'label_smoothing', 'best_val_acc', and per-epoch 'history'.
    """
    label_smoothing = model_training_params.label_smoothing
    device = torch.device(model_training_params.device)
    print(f"[{model_params.model}] Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_params.mean, std=data_params.std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_params.mean, std=data_params.std),
    ])

    batch_size = model_training_params.device_config.batch_size
    train_ds = datasets.CIFAR10(model_training_params.data_dir, train=True,  download=True, transform=transform_train)
    val_ds   = datasets.CIFAR10(model_training_params.data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=model_training_params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=model_training_params.num_workers)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_training_params.lr,
        weight_decay=model_training_params.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    ls_tag = f"_ls{label_smoothing}" if label_smoothing > 0.0 else ""
    stem, ext = os.path.splitext(model_training_params.save_path)
    model_save_path = f"{stem}_{model_params.model.lower()}{ls_tag}{ext}"

    best_acc     = 0.0
    best_weights = None
    patience_counter = 0
    history: list = []

    for epoch in range(1, model_training_params.epochs + 1):
        print(f"\n[{model_params.model}{ls_tag}] Epoch {epoch}/{model_training_params.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, model_training_params.log_interval
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss,  4),
            "train_acc":  round(tr_acc,   4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc,  4),
        })

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_save_path)
            patience_counter = 0
            print(f"  Saved best model → {model_save_path}  (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= model_training_params.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)

    results: Dict[str, Any] = {
        "model":           model_params.model,
        "label_smoothing": label_smoothing,
        "best_val_acc":    round(best_acc, 4),
        "history":         history,
    }
    results_save_path = model_save_path.replace(ext, "_results.json")
    with open(results_save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_params.model}{ls_tag}] Best val accuracy: {best_acc:.4f}")
    return model, results


def run_kd_training(
    student: nn.Module,
    teacher: nn.Module,
    student_params: Any,
    model_training_params: ModelTrainingConfig,
    data_params: Cifar10Params,
    modified_kd: bool = False,
    save_tag: str = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Knowledge distillation training loop (to be implemented in a later step).

    Args:
        student: Student model to train.
        teacher: Pretrained teacher model (frozen during distillation).
        student_params: Model-specific params dataclass for the student.
        model_training_params: ModelTrainingConfig with training hyperparameters.
        data_params: Cifar10Params with normalisation statistics.
        modified_kd: If True, uses modified KD where teacher probability is
            assigned only to the true class and remaining probability is spread
            equally across other classes.

    Returns:
        Tuple of (best_student, results).

    """
    device = torch.device(model_training_params.device)
    print(f"[{student_params.model}/{'modified_kd' if modified_kd else 'kd'}] Using device: {device}")
    T     = model_training_params.kd_temperature
    alpha = model_training_params.kd_alpha

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_params.mean, std=data_params.std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_params.mean, std=data_params.std),
    ])

    batch_size = model_training_params.device_config.batch_size
    train_ds = datasets.CIFAR10(model_training_params.data_dir, train=True,  download=True, transform=transform_train)
    val_ds   = datasets.CIFAR10(model_training_params.data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=model_training_params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=model_training_params.num_workers)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=model_training_params.lr,
        weight_decay=model_training_params.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    kd_tag = "modified_kd" if modified_kd else "kd"
    stem, ext = os.path.splitext(model_training_params.save_path)
    name_stem = save_tag if save_tag else student_params.model.lower()
    model_save_path = f"{stem}_{name_stem}_{kd_tag}{ext}"

    best_acc     = 0.0
    best_weights = None
    patience_counter = 0
    history: list = []

    for epoch in range(1, model_training_params.epochs + 1):
        print(f"\n[{student_params.model}/{kd_tag}] Epoch {epoch}/{model_training_params.epochs}")

        if modified_kd:
            tr_loss, tr_acc = train_one_epoch_teacher_prob(
                student, teacher, train_loader, optimizer, device,
                model_training_params.log_interval,
            )
        else:
            tr_loss, tr_acc = train_one_epoch_kd(
                student, teacher, train_loader, optimizer, ce_criterion, device,
                T, alpha, model_training_params.log_interval,
            )

        val_loss, val_acc = validate(student, val_loader, ce_criterion, device)
        scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss,  4),
            "train_acc":  round(tr_acc,   4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc,  4),
        })

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, model_save_path)
            patience_counter = 0
            print(f"  Saved best model → {model_save_path}  (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= model_training_params.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    student.load_state_dict(best_weights)

    results: Dict[str, Any] = {"model": student_params.model, "kd_type": kd_tag, "best_val_acc": round(best_acc, 4), "history": history}
    if not modified_kd:
        results["temperature"] = T
        results["alpha"]       = alpha
    results_save_path = model_save_path.replace(ext, "_results.json")
    with open(results_save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{student_params.model}/{kd_tag}] Best val accuracy: {best_acc:.4f}")
    return student, results


def run_pretrained_training(
    model: nn.Module,
    model_params: PartAParams,
    model_training_params: ModelTrainingConfig,
    data_params: Cifar10Params,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Training loop for PART_A transfer learning (pretrained ResNet-18).

    option='resize': images upsampled to 224x224, only FC layer trained (early layers frozen).
    option='modify': conv1 and maxpool adapted for 32x32, all layers fine-tuned.

    Args:
        model: Pre-built and device-placed ResNet-18 model.
        model_params: PartAParams instance containing option, input_size, and num_classes.
        model_training_params: ModelTrainingConfig with lr, epochs, patience, etc.
        data_params: Cifar10Params with mean and std normalisation statistics.

    Returns:
        Tuple of (best_model, results) where results contains 'option',
        'best_val_acc', and per-epoch 'history'.
    """
    device = torch.device(model_training_params.device)
    print(f"[PART_A/{model_params.option}] Using device: {device}")

    if model_params.input_size == 224:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])

    train_ds = datasets.CIFAR10(model_training_params.data_dir, train=True,  download=True, transform=transform_train)
    val_ds   = datasets.CIFAR10(model_training_params.data_dir, train=False, download=True, transform=transform_test)

    batch_size = model_training_params.device_config.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=model_training_params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=model_training_params.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_training_params.lr,
        weight_decay=model_training_params.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Derive separate save paths per option so the two runs don't overwrite each other
    stem, ext = os.path.splitext(model_training_params.save_path)
    model_save_path   = f"{stem}_parta_{model_params.option}{ext}"
    results_save_path = f"{stem}_parta_{model_params.option}_results.json"

    best_acc     = 0.0
    best_weights = None
    patience_counter = 0
    history = []

    for epoch in range(1, model_training_params.epochs + 1):
        print(f"\n[PART_A/{model_params.option}] Epoch {epoch}/{model_training_params.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, model_training_params.log_interval
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss,  4),
            "train_acc":  round(tr_acc,   4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc,  4),
        })

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_save_path)
            patience_counter = 0
            print(f"  Saved best model → {model_save_path}  (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= model_training_params.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)

    results: Dict[str, Any] = {"option": model_params.option, "best_val_acc": round(best_acc, 4), "history": history}
    with open(results_save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[PART_A/{model_params.option}] Results saved → {results_save_path}")
    print(f"[PART_A/{model_params.option}] Best val accuracy: {best_acc:.4f}")
    return model, results


def train_one_epoch_augmix(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_jsd: float,
    log_interval: int,
) -> Tuple[float, float]:
    """Run one AugMix training epoch with JSD consistency loss.

    Each batch from the AugMixDataset contains (clean, aug1, aug2, labels).
    The total loss is:
        L = CE(logits_clean, labels) + lambda_jsd * JSD(p_clean, p_aug1, p_aug2)

    Args:
        model:       Model to train.
        loader:      DataLoader backed by AugMixDataset.
        optimizer:   Optimiser instance.
        criterion:   CrossEntropyLoss (applied to clean logits only).
        device:      Computation device.
        lambda_jsd:  Weight on the JSD consistency term.
        log_interval: Batches between progress prints.

    Returns:
        Tuple of (mean_loss, accuracy) over the epoch (measured on clean inputs).
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (clean, aug1, aug2, labels) in enumerate(loader):
        clean, aug1, aug2, labels = (
            clean.to(device), aug1.to(device), aug2.to(device), labels.to(device)
        )

        optimizer.zero_grad()
        logits_clean = model(clean)
        logits_aug1  = model(aug1)
        logits_aug2  = model(aug2)

        ce   = criterion(logits_clean, labels)
        jsd  = jsd_loss(logits_clean, logits_aug1, logits_aug2)
        loss = ce + lambda_jsd * jsd

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * clean.size(0)
        correct    += logits_clean.argmax(1).eq(labels).sum().item()
        n          += clean.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def run_augmix_training(
    model: nn.Module,
    model_params: Any,
    model_training_params: ModelTrainingConfig,
    data_params: Cifar10Params,
    augmix_severity: int = 3,
    augmix_width: int = 3,
    augmix_depth: int = -1,
    lambda_jsd: float = 12.0,
    input_size: int = 32,
    save_tag: str = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Training loop with AugMix + JSD loss for CIFAR-10.

    Mirrors run_training but wraps the training set in AugMixDataset so each
    sample yields three views (clean, aug1, aug2). The JSD consistency loss is
    added on top of standard cross-entropy.  Checkpoints are saved with an
    '_augmix' suffix so they coexist with vanilla trained models.

    Supports both 32×32 (PART_B) and 224×224 (PART_A resize) input sizes via
    the input_size parameter; pre_transform and val transform are chosen accordingly.

    Args:
        model:               Model to train.
        model_params:        Model-specific params dataclass.
        model_training_params: ModelTrainingConfig with all training hyperparameters.
        data_params:         Cifar10Params with normalisation statistics.
        augmix_severity:     AugMix operation magnitude ∈ [1, 10].
        augmix_width:        Number of augmentation chains per sample.
        augmix_depth:        Chain length; -1 = random ∈ [1, 3].
        lambda_jsd:          Weight on JSD consistency loss term.
        input_size:          Spatial resolution fed to the model (32 or 224).
        save_tag:            Optional checkpoint name stem override. If None,
                             derived from model_params.model and label_smoothing.

    Returns:
        Tuple of (best_model, results).
    """
    label_smoothing = model_training_params.label_smoothing
    device = torch.device(model_training_params.device)
    tag = save_tag or model_params.model
    print(f"[{tag}/augmix] Using device: {device}")

    # PIL-level augmentation — pre_transform and val transform depend on input_size
    if input_size == 224:
        pre_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])
    else:
        pre_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=data_params.mean, std=data_params.std),
        ])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_params.mean, std=data_params.std),
    ])

    batch_size = model_training_params.device_config.batch_size
    # base dataset with no transform → raw PIL images
    base_train_ds = datasets.CIFAR10(
        model_training_params.data_dir, train=True, download=True, transform=None,
    )
    val_ds = datasets.CIFAR10(
        model_training_params.data_dir, train=False, download=True, transform=transform_val,
    )

    train_ds = AugMixDataset(
        base_train_ds, pre_transform, preprocess,
        severity=augmix_severity, width=augmix_width, depth=augmix_depth,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=model_training_params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=model_training_params.num_workers)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_training_params.lr,
        weight_decay=model_training_params.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    ls_tag = f"_ls{label_smoothing}" if label_smoothing > 0.0 else ""
    stem, ext = os.path.splitext(model_training_params.save_path)
    name_stem  = save_tag if save_tag else f"{model_params.model.lower()}{ls_tag}"
    model_save_path = f"{stem}_{name_stem}_augmix{ext}"

    best_acc, best_weights = 0.0, None
    patience_counter = 0
    history: list = []

    for epoch in range(1, model_training_params.epochs + 1):
        print(f"\n[{tag}{ls_tag}/augmix] Epoch {epoch}/{model_training_params.epochs}")
        tr_loss, tr_acc = train_one_epoch_augmix(
            model, train_loader, optimizer, criterion, device,
            lambda_jsd, model_training_params.log_interval,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss,  4),
            "train_acc":  round(tr_acc,   4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc,  4),
        })

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_save_path)
            patience_counter = 0
            print(f"  Saved best model → {model_save_path}  (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= model_training_params.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)

    results: Dict[str, Any] = {
        "model":           model_params.model,
        "augmix":          True,
        "label_smoothing": label_smoothing,
        "lambda_jsd":      lambda_jsd,
        "best_val_acc":    round(best_acc, 4),
        "history":         history,
    }
    results_save_path = model_save_path.replace(ext, "_results.json")
    with open(results_save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_params.model}{ls_tag}/augmix] Best val accuracy: {best_acc:.4f}")
    return model, results
