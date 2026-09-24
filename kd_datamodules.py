"""DataModule for Event → RGB knowledge distillation (Part 2, `train_event_distilled.py`).

Reads a ``rgb<TAB>event<TAB>wnid<TAB>stem`` TSV so that RGB and event samples
are paired 1:1 (no filesystem search). Produces two DataLoaders — one CE
branch with strong augmentation and one KD branch with weak augmentation —
that share a ``SyncedDistributedSampler`` so the two streams stay in lock-step.

.. note::
   With PyTorch Lightning + DDP, the Trainer must be constructed with
   ``use_distributed_sampler=False`` (or ``replace_sampler_ddp=False`` on older
   Lightning versions). Otherwise Lightning replaces the shared sampler and
   the CE/KD alignment breaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import ImageFolder

from dist_utils import build_dist_container, build_dist_teacher_from_cfg
from variant_utils import (
    SetEpochForSamplerCallback,
    SyncedDistributedSampler,
    build_class_to_idx_from_train_root,
    load_pairs_tsv,
    select_indices_by_class_ratio,
    tfm_rgb_strong,
    tfm_rgb_weak,
)

# SetEpochForSamplerCallback is re-exported for downstream scripts.
__all__ = ['KDDataModule', 'SetEpochForSamplerCallback']


class PairedEventRgbDataset(Dataset):
    """Yields paired event/RGB samples indexed by the TSV row order."""

    def __init__(
        self,
        dist_container,
        event_paths: List[str],
        rgb_paths: List[str],
        rgb_transform,
        class_to_idx: Dict[str, int],
    ):
        dl_dict = getattr(dist_container, 'dataloader', None)
        assert isinstance(dl_dict, dict) and dl_dict, 'dist_container.dataloader is not a dict'
        assert len(event_paths) == len(rgb_paths), 'event_paths and rgb_paths lengths differ'

        self.dist_dataset = next(iter(dl_dict.values())).dataset
        self.event_paths = event_paths
        self.rgb_paths = rgb_paths
        self.rgb_transform = rgb_transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.event_paths)

    def __getitem__(self, idx: int):
        ev_sample = self.dist_dataset[idx]
        x_event = ev_sample['input_data'] if isinstance(ev_sample, dict) else ev_sample[0]

        rgb_path = Path(self.rgb_paths[idx])
        if not rgb_path.is_file():
            raise FileNotFoundError(f'RGB missing: {rgb_path}')

        with Image.open(rgb_path).convert('RGB') as im:
            rgb = self.rgb_transform(im)

        y = torch.tensor(self.class_to_idx[rgb_path.parent.name], dtype=torch.long)
        return {'event': x_event, 'rgb': rgb, 'label': y}


class KDDataModule:
    """Glue between the DiST event pipeline and a standard RGB ImageNet loader."""

    def __init__(
        self,
        rgb_train_root: str,
        rgb_val_root: str,
        mapping_txt: str,
        dist_weight_path: str,
        paired_tsv_path: str,
        batch_size: int = 128,
        num_workers: int = 8,
        seed: int = 1,
        train_ratio: float = 1.0,
    ):
        self.rgb_train_root = rgb_train_root
        self.rgb_val_root = rgb_val_root
        self.mapping_txt = mapping_txt
        self.dist_weight_path = dist_weight_path
        self.paired_tsv_path = paired_tsv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_ratio = train_ratio

        self.teacher = None
        self.train_container = None
        self.train_event_paths: List[str] = []
        self.train_rgb_paths: List[str] = []
        self.class_to_idx: Optional[Dict[str, int]] = None
        self.train_ds_ce: Optional[Dataset] = None
        self.train_ds_kd: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self._shared_sampler: Optional[Sampler[int]] = None

    def setup(self, stage: Optional[str] = None):
        if self.paired_tsv_path is None:
            raise ValueError('paired_tsv_path must be provided')

        self.class_to_idx = build_class_to_idx_from_train_root(self.rgb_train_root)

        rgb_paths, event_paths, wnids, _ = load_pairs_tsv(self.paired_tsv_path)
        keep = select_indices_by_class_ratio(wnids, ratio=self.train_ratio, seed=self.seed)
        self.train_rgb_paths = [rgb_paths[i] for i in keep]
        self.train_event_paths = [event_paths[i] for i in keep]

        cfg_train, self.train_container = build_dist_container(
            self.train_event_paths,
            self.mapping_txt,
            self.dist_weight_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
            mode='train',
        )
        self.teacher = build_dist_teacher_from_cfg(cfg_train)

        self.train_ds_ce = PairedEventRgbDataset(
            self.train_container, self.train_event_paths, self.train_rgb_paths,
            rgb_transform=tfm_rgb_strong(), class_to_idx=self.class_to_idx,
        )
        self.train_ds_kd = PairedEventRgbDataset(
            self.train_container, self.train_event_paths, self.train_rgb_paths,
            rgb_transform=tfm_rgb_weak(), class_to_idx=self.class_to_idx,
        )
        self.val_ds = ImageFolder(self.rgb_val_root, transform=tfm_rgb_weak())

        self._shared_sampler = SyncedDistributedSampler(
            self.train_ds_ce, seed=self.seed, shuffle=True, drop_last=True,
        )

    def _loader(self, ds: Dataset, *, sampler=None, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self):
        assert self.train_ds_ce is not None and self.train_ds_kd is not None
        assert self._shared_sampler is not None
        return {
            'ce': self._loader(self.train_ds_ce, sampler=self._shared_sampler, drop_last=True),
            'kd': self._loader(self.train_ds_kd, sampler=self._shared_sampler, drop_last=True),
        }

    def val_dataloader(self):
        assert self.val_ds is not None
        return self._loader(self.val_ds)
