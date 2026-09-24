"""Train a ResNet34 teacher used by the KD ablation baselines.

Supports two modalities, selected via ``--modality``:

* ``gray`` — reads standard RGB images and converts them to 3-channel grayscale
  on the fly. Output checkpoint feeds ``train_gray_distilled.py`` (Gray Distilled).
* ``bw``   — reads pre-binarised images from a parallel ``bw_train_root`` tree
  (NEAREST interpolation everywhere). Output checkpoint feeds
  ``train_bw_distilled.py`` (B&W Distilled).

Plain cross-entropy training; no knowledge distillation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from variant_utils import (
    BWFromTSVDataset,
    RGBFromTSVDataset,
    SetEpochForSamplerCallback,
    SyncedDistributedSampler,
    build_class_to_idx_from_train_root,
    load_pairs_tsv,
    make_trainer,
    make_warmup_cosine_lambda,
    subsample_paths_by_class_ratio,
    tfm_bw_strong,
    tfm_bw_val,
    tfm_gray3_strong,
    tfm_gray3_weak,
)


CODE_ROOT = Path(__file__).resolve().parent.parent
torch.set_float32_matmul_precision('high')


class LitResNet34CE(L.LightningModule):
    def __init__(self, lr: float, weight_decay: float, warmup_epochs: int = 5):
        super().__init__()
        self.net = models.resnet34(weights=None, num_classes=1000)
        self.ce = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.ce(logits, y)
        acc1 = (logits.argmax(1) == y).float().mean()
        if stage == 'train':
            self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log('train/acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val/acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, make_warmup_cosine_lambda(self.hparams.warmup_epochs, self.trainer.max_epochs),
        )
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}


class TeacherDataModule(L.LightningDataModule):
    """Dispatches between the Gray and B&W training pipelines on a single ``modality`` flag."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self._sampler: Optional[Sampler[int]] = None

    def setup(self, stage=None):
        args = self.args
        paths, _, wnids, _ = load_pairs_tsv(args.paired_tsv)
        paths = subsample_paths_by_class_ratio(paths, wnids, ratio=args.train_ratio, seed=args.seed)

        if args.modality == 'gray':
            class_to_idx = build_class_to_idx_from_train_root(args.rgb_train_root)
            self.train_ds = RGBFromTSVDataset(paths, class_to_idx, tfm_gray3_strong())
            self.val_ds = datasets.ImageFolder(args.rgb_val_root, transform=tfm_gray3_weak())
        elif args.modality == 'bw':
            class_to_idx = build_class_to_idx_from_train_root(args.bw_train_root)
            self.train_ds = BWFromTSVDataset(
                paths, class_to_idx, tfm_bw_strong(), bw_root=args.bw_train_root,
            )
            self.val_ds = datasets.ImageFolder(args.bw_val_root, transform=tfm_bw_val())
        else:
            raise ValueError(f'Unknown modality: {args.modality!r}')

        self._sampler = SyncedDistributedSampler(
            self.train_ds, seed=args.seed, shuffle=True, drop_last=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.args.batch_size, sampler=self._sampler, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(self.args.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True, drop_last=False,
            persistent_workers=(self.args.num_workers > 0),
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train a ResNet34 teacher (Gray or B&W modality)')
    p.add_argument('--modality', choices=('gray', 'bw'), required=True,
                   help='gray: 3-channel grayscale from RGB / bw: pre-binarised B&W tree.')
    p.add_argument('--paired_tsv', default=str(CODE_ROOT / 'lists' / 'paired_train.tsv'))
    # Gray-only roots
    p.add_argument('--rgb_train_root', default='/datasets/imagenet/train')
    p.add_argument('--rgb_val_root', default='/datasets/imagenet/val')
    # B&W-only roots
    p.add_argument('--bw_train_root', default='/datasets/imagenet_bw/train')
    p.add_argument('--bw_val_root', default='/datasets/imagenet_bw/val')
    # Training
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--train_ratio', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1)
    # Runtime
    p.add_argument('--outdir', default='./runs')
    p.add_argument('--run_name', default=None,
                   help='Defaults to resnet34_<modality>_teacher.')
    p.add_argument('--precision', default='32-true')
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--strategy', default='auto')
    return p


def main():
    args = build_argparser().parse_args()
    if args.run_name is None:
        args.run_name = f'resnet34_{args.modality}_teacher'
    L.seed_everything(args.seed, workers=True)

    dm = TeacherDataModule(args)
    dm.setup()

    model = LitResNet34CE(lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.outdir, exist_ok=True)
    logger = CSVLogger(save_dir=args.outdir, name=args.run_name)

    ckpt = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename='{epoch:02d}-{val_acc1:.4f}',
        save_last=True,
        auto_insert_metric_name=False,
    )
    lrmon = LearningRateMonitor(logging_interval='epoch')
    epoch_cb = SetEpochForSamplerCallback(lambda: dm._sampler)

    trainer = make_trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        logger=logger,
        callbacks=[ckpt, lrmon, epoch_cb],
        strategy=args.strategy,
        precision=args.precision,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    main()
