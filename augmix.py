"""AugMix data augmentation (Hendrycks et al., 2020).

Paper: https://arxiv.org/abs/1912.02781

Each training sample produces three views: (clean, aug1, aug2).
The Jensen-Shannon Divergence (JSD) consistency loss penalises
prediction inconsistency across these views.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from typing import Callable


# ── Severity helper ────────────────────────────────────────────────────────────

def _magnitude(severity: int, low: float, high: float) -> float:
    """Map severity ∈ [1, 10] linearly to [low, high]."""
    return low + (high - low) * severity / 10.0


# ── PIL augmentation operations ────────────────────────────────────────────────

def _autocontrast(img: Image.Image, _: int) -> Image.Image:
    return ImageOps.autocontrast(img)

def _equalize(img: Image.Image, _: int) -> Image.Image:
    return ImageOps.equalize(img)

def _posterize(img: Image.Image, severity: int) -> Image.Image:
    return ImageOps.posterize(img, max(1, 4 - int(_magnitude(severity, 0, 3))))

def _solarize(img: Image.Image, severity: int) -> Image.Image:
    return ImageOps.solarize(img, int(_magnitude(severity, 256, 0)))

def _rotate(img: Image.Image, severity: int) -> Image.Image:
    deg = _magnitude(severity, 0, 30)
    return img.rotate(deg if np.random.random() > 0.5 else -deg, resample=Image.BILINEAR)

def _shear_x(img: Image.Image, severity: int) -> Image.Image:
    s = _magnitude(severity, 0, 0.3)
    s = s if np.random.random() > 0.5 else -s
    return img.transform(img.size, Image.AFFINE, (1, s, 0, 0, 1, 0), Image.BILINEAR)

def _shear_y(img: Image.Image, severity: int) -> Image.Image:
    s = _magnitude(severity, 0, 0.3)
    s = s if np.random.random() > 0.5 else -s
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, s, 1, 0), Image.BILINEAR)

def _translate_x(img: Image.Image, severity: int) -> Image.Image:
    p = int(_magnitude(severity, 0, img.size[0] * 0.33))
    p = p if np.random.random() > 0.5 else -p
    return img.transform(img.size, Image.AFFINE, (1, 0, p, 0, 1, 0), Image.BILINEAR)

def _translate_y(img: Image.Image, severity: int) -> Image.Image:
    p = int(_magnitude(severity, 0, img.size[1] * 0.33))
    p = p if np.random.random() > 0.5 else -p
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, p), Image.BILINEAR)

def _color(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Color(img).enhance(_magnitude(severity, 0.1, 1.9))

def _brightness(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(_magnitude(severity, 0.1, 1.9))

def _contrast(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(_magnitude(severity, 0.1, 1.9))

def _sharpness(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(_magnitude(severity, 0.1, 1.9))


AUGMENTATIONS = [
    _autocontrast, _equalize, _posterize, _solarize,
    _rotate, _shear_x, _shear_y, _translate_x, _translate_y,
    _color, _brightness, _contrast, _sharpness,
]


# ── Core AugMix mixing ─────────────────────────────────────────────────────────

def _augmix_single(
    img: Image.Image,
    preprocess: Callable,
    severity: int,
    width: int,
    depth: int,
    alpha: float,
) -> torch.Tensor:
    """Produce one AugMix-augmented tensor from a PIL image.

    Samples `width` augmentation chains, mixes them with Dirichlet weights,
    then interpolates with the clean image using a Beta-sampled coefficient.

    Args:
        img:        Pre-processed PIL image (after crop/flip).
        preprocess: ToTensor + Normalize callable.
        severity:   Augmentation magnitude ∈ [1, 10].
        width:      Number of parallel augmentation chains.
        depth:      Chain length; -1 → random in [1, 3].
        alpha:      Dirichlet / Beta concentration parameter.

    Returns:
        Float tensor with same shape as preprocess(img).
    """
    ws  = np.float32(np.random.dirichlet([alpha] * width))
    m   = np.float32(np.random.beta(alpha, alpha))
    mix = torch.zeros_like(preprocess(img))
    for i in range(width):
        img_aug = img.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = AUGMENTATIONS[np.random.randint(len(AUGMENTATIONS))]
            img_aug = op(img_aug, severity)
        mix = mix + ws[i] * preprocess(img_aug)
    return (1.0 - m) * preprocess(img) + m * mix


# ── Dataset wrapper ────────────────────────────────────────────────────────────

class AugMixDataset(Dataset):
    """Wraps a torchvision dataset and returns (clean, aug1, aug2, label) tuples.

    The base dataset must be created with transform=None so that __getitem__
    yields raw PIL images.  Two transforms are applied in sequence:

    1. pre_transform  — PIL-level spatial augmentation (RandomCrop, RandomHorizontalFlip).
    2. preprocess     — ToTensor + Normalize.

    AugMix operates between these two steps, applying random operation chains to
    the spatially-augmented PIL image before normalisation.

    Args:
        base_dataset:  torchvision dataset with transform=None.
        pre_transform: PIL → PIL transform (crop / flip).
        preprocess:    PIL → Tensor transform (ToTensor + Normalize).
        severity:      Augmentation magnitude ∈ [1, 10]. Default 3.
        width:         Number of augmentation chains. Default 3.
        depth:         Chain length; -1 = random ∈ [1, 3]. Default -1.
        alpha:         Dirichlet/Beta concentration. Default 1.0.
    """

    def __init__(
        self,
        base_dataset,
        pre_transform: Callable,
        preprocess: Callable,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
    ) -> None:
        self.dataset       = base_dataset
        self.pre_transform = pre_transform
        self.preprocess    = preprocess
        self.severity      = severity
        self.width         = width
        self.depth         = depth
        self.alpha         = alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]          # raw PIL image
        img   = self.pre_transform(img)         # crop + flip  →  PIL
        clean = self.preprocess(img)            # PIL → tensor
        aug1  = _augmix_single(img, self.preprocess, self.severity, self.width, self.depth, self.alpha)
        aug2  = _augmix_single(img, self.preprocess, self.severity, self.width, self.depth, self.alpha)
        return clean, aug1, aug2, label


# ── JSD consistency loss ───────────────────────────────────────────────────────

def jsd_loss(
    logits_clean: torch.Tensor,
    logits_aug1: torch.Tensor,
    logits_aug2: torch.Tensor,
) -> torch.Tensor:
    """Jensen-Shannon Divergence consistency loss (Hendrycks et al., 2020).

    JSD(p_clean ‖ p_aug1 ‖ p_aug2) = (1/3) Σ_i KL(p_i ‖ p_mixture)
    where  p_mixture = (p_clean + p_aug1 + p_aug2) / 3.

    Args:
        logits_clean: [B, C] logits for clean images.
        logits_aug1:  [B, C] logits for first AugMix view.
        logits_aug2:  [B, C] logits for second AugMix view.

    Returns:
        Scalar JSD loss tensor.
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1  = F.softmax(logits_aug1,  dim=1)
    p_aug2  = F.softmax(logits_aug2,  dim=1)
    p_mix   = ((p_clean + p_aug1 + p_aug2) / 3.0).clamp(1e-7, 1.0)
    log_mix = p_mix.log()
    return (
        F.kl_div(log_mix, p_clean, reduction="batchmean") +
        F.kl_div(log_mix, p_aug1,  reduction="batchmean") +
        F.kl_div(log_mix, p_aug2,  reduction="batchmean")
    ) / 3.0
