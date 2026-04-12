"""PGD adversarial attack for CIFAR-10 models.

Reference: Madry et al., "Towards Deep Learning Models Resistant to
Adversarial Attacks," ICLR 2018. https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    return x * _STD.to(x.device) + _MEAN.to(x.device)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str = "linf",
    eps: float = 4 / 255,
    alpha: float = 1 / 255,
    num_steps: int = 20,
) -> torch.Tensor:
    """PGD attack supporting both L∞ and L2 norm constraints.

    Operates in pixel space [0, 1]; input and output tensors are in the
    model's normalised space.  For L∞ uses a random start; for L2 starts
    from the clean image.

    Args:
        model:     Target model (should be in eval mode).
        x:         Normalised input images [B, C, H, W].
        y:         True class labels [B].
        norm:      Constraint type: ``'linf'`` or ``'l2'``.
        eps:       Perturbation budget in pixel space
                   (default 4/255 for L∞, 0.25 for L2).
        alpha:     Step size in pixel space
                   (default 1/255 for L∞, 0.05 for L2).
        num_steps: Number of PGD iterations (default 20).

    Returns:
        Adversarial examples in the same normalised space as *x*.

    Raises:
        ValueError: If *norm* is not ``'linf'`` or ``'l2'``.
    """
    if norm not in ("linf", "l2"):
        raise ValueError(f"norm must be 'linf' or 'l2', got '{norm}'")

    model.eval()
    x_pix = _denorm(x).clamp(0.0, 1.0)

    if norm == "linf":
        delta = torch.empty_like(x_pix).uniform_(-eps, eps)
        x_adv = (x_pix + delta).clamp(0.0, 1.0).detach()
    else:  # l2 — small random start on the epsilon sphere
        delta = torch.randn_like(x_pix)
        d_norm = delta.view(delta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
        delta = delta / (d_norm + 1e-8) * (eps * 0.01)
        x_adv = (x_pix + delta).clamp(0.0, 1.0).detach()

    for _ in range(num_steps):
        x_adv = x_adv.requires_grad_(True)
        loss  = F.cross_entropy(model(_norm(x_adv)), y)
        loss.backward()

        with torch.no_grad():
            if norm == "linf":
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.max(torch.min(x_adv, x_pix + eps), x_pix - eps)
            else:  # l2
                grad  = x_adv.grad
                g_norm = grad.view(grad.shape[0], -1).norm(2, dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
                x_adv  = x_adv + alpha * (grad / g_norm)
                delta  = x_adv - x_pix
                d_norm = delta.view(delta.shape[0], -1).norm(2, dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
                x_adv  = x_pix + delta * (eps / d_norm).clamp(max=1.0)

            x_adv = x_adv.clamp(0.0, 1.0)

    return _norm(x_adv)
