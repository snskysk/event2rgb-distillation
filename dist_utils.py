"""Helpers for plugging the DiST event-domain ResNet34 teacher into the KD pipeline.

The module builds a ``train_event_teacher.data.ImageNetContainer`` /
``train_event_teacher.models.CNNContainer`` pair from a flat list of event
file paths, so the KD DataModule can iterate over events and RGB images in
lock-step with a single index.
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Make ``code_submittion/`` importable so ``train_event_teacher`` resolves
# regardless of the directory this script is launched from.
_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from train_event_teacher.data.data_container import ImageNetContainer
from train_event_teacher.models.model_container import CNNContainer

# NumPy 2.x compatibility shim for downstream code that still references ``np.float``.
import numpy as _np
if not hasattr(_np, 'float'):
    _np.float = float


# ---------------------------------------------------------------------------
# Event-path collection
# ---------------------------------------------------------------------------

def collect_event_paths_train_all(root_dir: str, exts: Tuple[str, ...] = ('.npz',)) -> List[str]:
    """Enumerate every event file under ``{root_dir}/Part_*/<wnid>/``.

    Matches the N-ImageNet training layout ``/.../training/Part_*/<wnid>/*.npz``.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'train event_root not found: {root_dir}')
    part_dirs = sorted(p for p in root.glob('Part_*') if p.is_dir())
    if not part_dirs:
        raise RuntimeError(f'No Part_* directories under: {root_dir}')
    paths: List[str] = []
    for part in part_dirs:
        for wnid_dir in sorted(d for d in part.iterdir() if d.is_dir()):
            for ext in exts:
                paths.extend(str(p) for p in wnid_dir.glob(f'*{ext}'))
    if not paths:
        raise RuntimeError(f'No event files found under training root: {root_dir}')
    return sorted(paths)


def collect_event_paths_val(root_dir: str, exts: Tuple[str, ...] = ('.npz',)) -> List[str]:
    """Enumerate event files under ``{root_dir}/<wnid>/`` (e.g. ``validation/extracted_val``)."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'val event_root not found: {root_dir}')
    paths: List[str] = []
    for wnid_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for ext in exts:
            paths.extend(str(p) for p in wnid_dir.glob(f'*{ext}'))
    if not paths:
        raise RuntimeError(f'No event files found under val root: {root_dir}')
    return sorted(paths)


# ---------------------------------------------------------------------------
# Per-class subsampling (ceil-rounded, at least one sample per class)
# ---------------------------------------------------------------------------

def subsample_by_class_ratio(event_paths: List[str], ratio: float, seed: int = 1) -> List[str]:
    """Keep ``ceil(n * ratio)`` events per WNID (at least one), shuffled with ``seed``."""
    if ratio <= 0:
        raise ValueError('ratio must be > 0')
    ratio = min(float(ratio), 1.0)
    random.seed(seed)

    buckets: Dict[str, List[str]] = defaultdict(list)
    for p in event_paths:
        buckets[Path(p).parent.name].append(p)

    picked: List[str] = []
    for files in buckets.values():
        files = sorted(files)
        k = max(1, math.ceil(len(files) * ratio))
        picked.extend(files if k >= len(files) else random.sample(files, k))

    random.seed(seed)
    random.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------
# Lightweight config object consumed by DiST's ImageNetContainer / CNNContainer
# ---------------------------------------------------------------------------

class CfgObj:
    # General
    name = 'dist_kd'
    mode = 'test'
    height = 224
    width = 224
    no_cuda = False
    batch_size = 256
    seed = 1
    parallel = True
    model = 'ResNet34'
    channel_size = 2
    kernel_size = 14
    num_classes = 1000
    save_root_dir = './experiments'

    # DataLoader
    pin_memory = True
    num_workers = 6
    augment = False
    augment_type = 'base_augment'
    on_the_fly = False
    quantized = True
    trim_class_count = None
    collate_type = 'normal'

    # Preprocessing
    reshape = True
    reshape_method = 'no_sample'
    interpolate = None
    loader_type = 'reshape_then_acc_adj_sort'
    parser_type = 'normal'

    # Slice
    slice_events = True
    slice_length = 30000
    slice_method = 'random'
    slice_augment = False
    slice_augment_width = 0
    slice_start = 0
    slice_end = 30000

    # Feature flags consumed by DiST
    neglect_polarity = False
    global_time = True
    strict = False
    use_image = False
    denoise_sort = False
    denoise_image = False
    filter_flash = False
    filter_noise = False
    quantize_sort = None

    # Required by CNNContainer
    pretrained = False
    pretrained_num_classes = 1000
    freeze = False
    keep_fc = True
    train_classifier = False

    # Unused but required by the trainer abstractions
    optimizer = 'Adam'
    learning_rate = 3e-4
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 0
    save_every = 1

    # Runtime fields
    label_map: str = None
    train_file: str = None
    val_file: str = None
    load_model: str = None

    def __init__(
        self,
        label_map: str,
        list_path: str,
        load_model: str,
        batch_size: int = 256,
        num_workers: int = 6,
        seed: int = 1,
        mode: str = 'test',
    ):
        self.label_map = label_map
        self.train_file = list_path
        self.val_file = list_path
        self.load_model = load_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.mode = mode


def _write_list_tmp(paths: List[str]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    with open(tmp.name, 'w') as f:
        f.write('\n'.join(paths) + '\n')
    return tmp.name


def robust_load_state_into_model(model: torch.nn.Module, weight_path: str) -> None:
    state = torch.load(weight_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[Teacher load] missing={len(missing)}, unexpected={len(unexpected)}')


def build_dist_container(
    event_paths: List[str],
    mapping_txt: str,
    dist_weight_path: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    mode: str,
) -> Tuple[CfgObj, ImageNetContainer]:
    """Materialise a DiST ``ImageNetContainer`` over the provided event paths."""
    list_path = _write_list_tmp(event_paths)
    cfg = CfgObj(
        mapping_txt, list_path, dist_weight_path,
        batch_size=batch_size, num_workers=num_workers, seed=seed, mode=mode,
    )
    return cfg, ImageNetContainer(cfg)


def build_dist_teacher_from_cfg(cfg: CfgObj) -> torch.nn.Module:
    """Instantiate the DiST ResNet34 teacher, load the checkpoint and freeze it."""
    model_container = CNNContainer(cfg)
    teacher = model_container.models['model']
    if cfg.load_model:
        robust_load_state_into_model(teacher, cfg.load_model)
        cfg.load_model = None
    cfg.pretrained_num_classes = None  # prevents a downstream FC overwrite
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher
