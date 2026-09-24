"""Shared utilities used across Part 2 training scripts in ``resnet34_variant/``.

The scripts in ``resnet34_variant/`` trained multiple ResNet34 variants
(Event-Distilled, Self-Distilled, Gray-Distilled, B&W-Distilled, plus the
adversarially and Gaussian-noise-trained baselines). They previously held
large chunks of copy-pasted code (TSV loader, DDP sampler, transforms,
checkpoint loader, LR schedule, ...). This module consolidates them so that
each training script can focus on its distillation-specific details.
"""

from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# TSV utilities
# ---------------------------------------------------------------------------

def load_pairs_tsv(tsv_path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Read a ``rgb<TAB>event<TAB>wnid<TAB>stem`` TSV file."""
    rgb_paths: List[str] = []
    event_paths: List[str] = []
    wnids: List[str] = []
    stems: List[str] = []
    with open(tsv_path, 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f'Bad TSV line (NF!=4): {row}')
            rgb, event, wnid, stem = row
            rgb_paths.append(rgb)
            event_paths.append(event)
            wnids.append(wnid)
            stems.append(stem)
    return rgb_paths, event_paths, wnids, stems


def build_class_to_idx_from_train_root(rgb_train_root: str) -> Dict[str, int]:
    """Replicate ``torchvision.datasets.ImageFolder.class_to_idx`` from a root directory."""
    root = Path(rgb_train_root)
    wnids = sorted(p.name for p in root.iterdir() if p.is_dir())
    return {wnid: i for i, wnid in enumerate(wnids)}


def select_indices_by_class_ratio(wnids: List[str], ratio: float, seed: int) -> List[int]:
    """Return a deterministic index subset so each class retains ``ceil(n * ratio)`` items.

    The returned indices are shuffled with ``seed``; pass them to any parallel list
    (e.g. ``rgb_paths`` and ``event_paths`` at once) to keep the alignment consistent.
    """
    if ratio >= 1.0:
        return list(range(len(wnids)))
    buckets: Dict[str, List[int]] = {}
    for i, w in enumerate(wnids):
        buckets.setdefault(w, []).append(i)
    rng = random.Random(seed)
    keep: List[int] = []
    for idxs in buckets.values():
        k = max(1, int(math.ceil(len(idxs) * ratio)))
        keep.extend(idxs if k >= len(idxs) else rng.sample(idxs, k))
    rng.shuffle(keep)
    return keep


def subsample_paths_by_class_ratio(
    paths: List[str],
    wnids: List[str],
    ratio: float,
    seed: int,
) -> List[str]:
    """Convenience wrapper over :func:`select_indices_by_class_ratio` for a single list."""
    if ratio >= 1.0:
        return paths
    keep = select_indices_by_class_ratio(wnids, ratio, seed)
    return [paths[i] for i in keep]


# ---------------------------------------------------------------------------
# Distributed-aware sampler with CE/KD synchronisation
# ---------------------------------------------------------------------------

class SyncedDistributedSampler(Sampler[int]):
    """Sampler that produces an identical permutation for every rank using ``seed + epoch``.

    Two DataLoaders that share the same instance will yield indices in lock-step,
    which is required when CE- and KD-branch samples must stay aligned within a step.
    Rank/world size is re-queried on every ``__iter__``, so the sampler can be
    constructed before DDP is initialised.
    """

    def __init__(self, dataset: Dataset, seed: int = 0, shuffle: bool = True, drop_last: bool = True):
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.num_replicas = 1
        self.rank = 0

    def _refresh_dist(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

    def _compute_sizes(self, n: int) -> Tuple[int, int]:
        if self.drop_last:
            num_samples = n // self.num_replicas
            total_size = num_samples * self.num_replicas
        else:
            num_samples = int(math.ceil(n / self.num_replicas))
            total_size = num_samples * self.num_replicas
        return num_samples, total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        self._refresh_dist()
        n = len(self.dataset)
        _, total_size = self._compute_sizes(n)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.drop_last:
            indices = indices[:total_size]
        elif len(indices) < total_size:
            indices += indices[: (total_size - len(indices))]

        indices = indices[self.rank: total_size: self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        self._refresh_dist()
        num_samples, _ = self._compute_sizes(len(self.dataset))
        return num_samples


class SetEpochForSamplerCallback(L.Callback):
    """Call ``sampler.set_epoch(epoch)`` at the start of each training epoch."""

    def __init__(self, get_sampler_fn: Callable[[], Optional[Sampler]]):
        super().__init__()
        self.get_sampler_fn = get_sampler_fn

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: D401, ARG002
        sampler = self.get_sampler_fn()
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(trainer.current_epoch)


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _normalize() -> T.Normalize:
    return T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))


def tfm_rgb_strong(*, randaugment: bool = True, random_erasing: bool = True) -> T.Compose:
    """Strong RGB augmentation used for the student's CE branch."""
    ops = [
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
    ]
    if randaugment:
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    ops += [T.ToTensor(), _normalize()]
    if random_erasing:
        ops.append(T.RandomErasing(p=0.25))
    return T.Compose(ops)


def tfm_rgb_weak() -> T.Compose:
    """Weak RGB transform used for the KD branch and for validation."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        _normalize(),
    ])


def tfm_gray3_weak() -> T.Compose:
    """Weak 3-channel grayscale transform for the Gray-Distilled teacher."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        _normalize(),
    ])


def tfm_gray3_strong() -> T.Compose:
    """Strong 3-channel grayscale transform for training a Gray teacher from scratch."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        _normalize(),
        T.RandomErasing(p=0.25),
    ])


def tfm_bw_weak() -> T.Compose:
    """B&W teacher input. NEAREST interpolation preserves the binary structure."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.NEAREST),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        _normalize(),
    ])


def tfm_bw_strong() -> T.Compose:
    """Strong B&W transform for training a B&W teacher from scratch (NEAREST everywhere)."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.NEAREST),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        _normalize(),
    ])


def tfm_bw_val() -> T.Compose:
    """Validation transform for a binarised (B&W) dataset."""
    return T.Compose([
        T.Resize(256, interpolation=InterpolationMode.NEAREST),
        T.CenterCrop(224),
        T.ToTensor(),
        _normalize(),
    ])


class AddGaussianNoise(nn.Module):
    """Transform that adds zero-mean Gaussian noise to a tensor image with probability ``p``."""

    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 0.5):
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.p = float(p)

    def forward(self, img: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            img = torch.clamp(img + noise, 0.0, 1.0)
        return img


# ---------------------------------------------------------------------------
# Dataset classes for the shared TSV layout
# ---------------------------------------------------------------------------

def _label_from_path(path: Path, class_to_idx: Dict[str, int]) -> torch.Tensor:
    return torch.tensor(class_to_idx[path.parent.name], dtype=torch.long)


class RGBFromTSVDataset(Dataset):
    """Single-transform RGB dataset driven by a list of paths."""

    def __init__(self, rgb_paths: List[str], class_to_idx: Dict[str, int], transform):
        self.rgb_paths = rgb_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        p = Path(self.rgb_paths[idx])
        if not p.is_file():
            raise FileNotFoundError(f'RGB missing: {p}')
        y = _label_from_path(p, self.class_to_idx)
        with Image.open(p).convert('RGB') as im:
            x = self.transform(im)
        return x, y


class RGBFromTSVDictDataset(Dataset):
    """Same as ``RGBFromTSVDataset`` but yields a dict (matches the KD collate format)."""

    def __init__(self, rgb_paths: List[str], class_to_idx: Dict[str, int], transform):
        self.rgb_paths = rgb_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        p = Path(self.rgb_paths[idx])
        if not p.is_file():
            raise FileNotFoundError(f'RGB missing: {p}')
        y = _label_from_path(p, self.class_to_idx)
        with Image.open(p).convert('RGB') as im:
            x = self.transform(im)
        return {'rgb': x, 'label': y}


class RgbAndGrayDataset(Dataset):
    """Yields ``{'rgb': ..., 'gray': ..., 'label': ...}`` from a single image open.

    Used by the Gray-Distilled KD branch where the teacher sees a 3-channel
    grayscale view and the student sees the weakly-augmented RGB view.
    """

    def __init__(self, rgb_paths: List[str], class_to_idx: Dict[str, int], rgb_tfm, gray_tfm):
        self.rgb_paths = rgb_paths
        self.class_to_idx = class_to_idx
        self.rgb_tfm = rgb_tfm
        self.gray_tfm = gray_tfm

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        p = Path(self.rgb_paths[idx])
        if not p.is_file():
            raise FileNotFoundError(f'RGB missing: {p}')
        y = _label_from_path(p, self.class_to_idx)
        with Image.open(p).convert('RGB') as im:
            x_rgb = self.rgb_tfm(im)
            x_gray = self.gray_tfm(im)
        return {'rgb': x_rgb, 'gray': x_gray, 'label': y}


class PairedRgbBwDataset(Dataset):
    """Returns the three tensors required by the B&W-Distilled KD branch.

    Keys: ``rgb_ce`` (strong-aug RGB for the CE loss), ``rgb_kd`` (weak-aug RGB fed
    to the student in the KD loss) and ``bw_kd`` (weak-aug binarised image fed
    to the teacher). If the precomputed B&W file is missing, we fall back to
    Otsu binarisation applied on the fly.
    """

    _IMG_EXTS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')

    def __init__(
        self,
        rgb_paths: List[str],
        class_to_idx: Dict[str, int],
        bw_root: str,
        tfm_ce,
        tfm_kd,
        tfm_bw,
    ):
        self.rgb_paths = rgb_paths
        self.class_to_idx = class_to_idx
        self.bw_root = Path(bw_root)
        self.tfm_ce = tfm_ce
        self.tfm_kd = tfm_kd
        self.tfm_bw = tfm_bw

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def _find_bw(self, p_rgb: Path) -> Optional[Path]:
        bw_dir = self.bw_root / p_rgb.parent.name
        for ext in self._IMG_EXTS:
            candidate = bw_dir / f'{p_rgb.stem}{ext}'
            if candidate.is_file():
                return candidate
        return None

    def __getitem__(self, idx: int):
        p_rgb = Path(self.rgb_paths[idx])
        if not p_rgb.is_file():
            raise FileNotFoundError(f'RGB missing: {p_rgb}')

        p_bw = self._find_bw(p_rgb)
        y = _label_from_path(p_rgb, self.class_to_idx)

        with Image.open(p_rgb).convert('RGB') as im_rgb:
            x_rgb_ce = self.tfm_ce(im_rgb)
            x_rgb_kd = self.tfm_kd(im_rgb)
            gray_np = None if p_bw is not None else _pil_to_gray_np(im_rgb)

        if p_bw is not None:
            with Image.open(p_bw).convert('RGB') as im_bw:
                x_bw_kd = self.tfm_bw(im_bw)
        else:
            x_bw_kd = self.tfm_bw(_otsu_bw_pil(gray_np))

        return {'rgb_ce': x_rgb_ce, 'rgb_kd': x_rgb_kd, 'bw_kd': x_bw_kd, 'label': y}


def _pil_to_gray_np(im: Image.Image):
    import numpy as np
    return np.array(im.convert('L'))


def _otsu_bw_pil(gray_np) -> Image.Image:
    import cv2
    _, bw_np = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(bw_np).convert('RGB')


class BWFromTSVDataset(Dataset):
    """TSV-driven dataset that remaps each RGB path onto a parallel B&W tree."""

    def __init__(self, original_paths: List[str], class_to_idx: Dict[str, int], transform, bw_root: str):
        self.original_paths = original_paths
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.bw_root = Path(bw_root)

    def __len__(self) -> int:
        return len(self.original_paths)

    def __getitem__(self, idx: int):
        orig_p = Path(self.original_paths[idx])
        p = self.bw_root / orig_p.parent.name / orig_p.name
        if not p.is_file():
            raise FileNotFoundError(f'B&W image missing: {p}')
        y = _label_from_path(p, self.class_to_idx)
        with Image.open(p).convert('RGB') as im:
            x = self.transform(im)
        return x, y


# ---------------------------------------------------------------------------
# Teacher checkpoint loading (Lightning-style .ckpt files)
# ---------------------------------------------------------------------------

def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ('state_dict', 'model', 'net', 'model_state', 'state_dict_ema', 'ema'):
            inner = ckpt.get(k)
            if isinstance(inner, dict):
                return inner
        return ckpt  # type: ignore[return-value]
    raise ValueError('ckpt is not a dict-like object')


def _strip_prefixes_repeat(key: str, prefixes: Tuple[str, ...]) -> str:
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


_GENERIC_STRIP = ('state_dict.', 'module.', 'model.', 'net.', 'backbone.', 'resnet.', 'vgg.', 'convnext.')
_STUDENT_PREFIXES = ('student.', 'model.student.', 'net.student.', 'student.net.', 'module.student.')


def load_resnet34_from_ckpt(
    ckpt_path: Optional[str],
    *,
    num_classes: int = 1000,
    prefer_student: bool = True,
    freeze: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Load a torchvision ResNet34 whose weights may sit under various Lightning prefixes.

    Keys starting with ``teacher.`` are dropped. If ``prefer_student`` is true and the
    checkpoint contains any ``student.*`` keys, they are remapped to the model root;
    otherwise generic wrapper prefixes (``module.``, ``model.``, ...) are stripped.
    """
    model = models.resnet34(weights=None, num_classes=num_classes)
    if not ckpt_path:
        return _maybe_freeze(model, freeze)
    if not os.path.exists(ckpt_path):
        if verbose:
            print(f'Warning: checkpoint not found at {ckpt_path}. Returning randomly-initialised model.')
        return _maybe_freeze(model, freeze)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd_raw = _extract_state_dict(ckpt)
    has_student = prefer_student and any(
        any(k.startswith(p) for p in _STUDENT_PREFIXES) for k in sd_raw.keys()
    )

    target = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in sd_raw.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith('teacher.') or '.teacher.' in k:
            continue
        k2 = k
        if has_student and any(k2.startswith(p) for p in _STUDENT_PREFIXES):
            k2 = _strip_prefixes_repeat(k2, _STUDENT_PREFIXES)
        k2 = _strip_prefixes_repeat(k2, _GENERIC_STRIP)
        if k2 in target and target[k2].shape == v.shape:
            filtered[k2] = v

    model.load_state_dict(filtered, strict=False)
    if verbose:
        print(f'Loaded {len(filtered)}/{len(target)} parameters from {ckpt_path}')
    return _maybe_freeze(model, freeze)


def _maybe_freeze(model: nn.Module, freeze: bool) -> nn.Module:
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    return model


# ---------------------------------------------------------------------------
# KD loss and LR schedule helpers
# ---------------------------------------------------------------------------

def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard temperature-scaled KL divergence loss used throughout the paper."""
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction='batchmean',
    ) * (t * t)


def make_trainer(
    *,
    max_epochs: int,
    devices,
    logger,
    callbacks,
    strategy: str = 'auto',
    precision: str = '32-true',
    deterministic: bool = True,
    log_every_n_steps: int = 50,
) -> 'L.Trainer':
    """Build a ``L.Trainer`` with ``use_distributed_sampler=False`` (required for the synced sampler)."""
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        deterministic=deterministic,
        use_distributed_sampler=False,
    )


def make_warmup_cosine_lambda(warmup_epochs: int, max_epochs: int) -> Callable[[int], float]:
    """Linear warm-up for ``warmup_epochs`` followed by a half-cosine decay to 0."""
    warmup_epochs = max(1, int(warmup_epochs))
    max_epochs = max(warmup_epochs + 1, int(max_epochs))

    def lr_lambda(epoch: int) -> float:
        e = epoch + 1
        if e <= warmup_epochs:
            return e / warmup_epochs
        t = (e - warmup_epochs) / (max_epochs - warmup_epochs)
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return lr_lambda
