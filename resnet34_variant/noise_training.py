"""Gaussian-noise training baseline used for the "Noise Training" row in the paper.

Adds zero-mean Gaussian noise to each training image with probability ``p`` and
trains a plain CE ResNet34 on top of a pretrained checkpoint.
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
from torchvision import datasets
from torchvision.transforms import v2 as T

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from variant_utils import (
    AddGaussianNoise,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RGBFromTSVDataset,
    SetEpochForSamplerCallback,
    SyncedDistributedSampler,
    build_class_to_idx_from_train_root,
    load_pairs_tsv,
    load_resnet34_from_ckpt,
    make_trainer,
    subsample_paths_by_class_ratio,
)


torch.set_float32_matmul_precision('high')
torch.use_deterministic_algorithms(False)


def tfm_train_noisy(noise_std: float, noise_p: float) -> T.Compose:
    """Strong augmentation with in-the-loop Gaussian noise injection."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        AddGaussianNoise(mean=0.0, std=noise_std, p=noise_p),  # inserted before Normalize
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        T.RandomErasing(p=0.25),
    ])


def tfm_val() -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


class LitResNet34CENoise(L.LightningModule):
    def __init__(self, lr: float, weight_decay: float, ckpt_path: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        self.net = load_resnet34_from_ckpt(ckpt_path)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce(logits, y)
        acc1 = (logits.argmax(1) == y).float().mean()
        self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce(logits, y)
        acc1 = (logits.argmax(1) == y).float().mean()
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': sch}


class RGBDataModuleFromTSV(L.LightningDataModule):
    def __init__(self, rgb_train_root, rgb_val_root, paired_tsv_path,
                 ratio, seed, batch_size, num_workers, noise_std, noise_p):
        super().__init__()
        self.rgb_train_root = rgb_train_root
        self.rgb_val_root = rgb_val_root
        self.paired_tsv_path = paired_tsv_path
        self.ratio = ratio
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_std = noise_std
        self.noise_p = noise_p
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self._sampler: Optional[Sampler[int]] = None

    def setup(self, stage=None):
        class_to_idx = build_class_to_idx_from_train_root(self.rgb_train_root)
        rgb_paths, _, wnids, _ = load_pairs_tsv(self.paired_tsv_path)
        rgb_paths = subsample_paths_by_class_ratio(rgb_paths, wnids, ratio=self.ratio, seed=self.seed)

        self.train_ds = RGBFromTSVDataset(
            rgb_paths, class_to_idx, tfm_train_noisy(self.noise_std, self.noise_p),
        )
        self.val_ds = datasets.ImageFolder(self.rgb_val_root, transform=tfm_val())
        self._sampler = SyncedDistributedSampler(self.train_ds, seed=self.seed, shuffle=True, drop_last=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, sampler=self._sampler, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Gaussian-noise training baseline')
    p.add_argument('--rgb_train_root', default='/datasets/imagenet/train')
    p.add_argument('--rgb_val_root', default='/datasets/imagenet/val')
    p.add_argument('--paired_tsv', default=str(CODE_ROOT / 'lists' / 'paired_train.tsv'))
    p.add_argument('--pretrained_ckpt', default=None)
    # Noise hyperparameters
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--noise_p', type=float, default=0.5)
    # Training
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=2e-4)
    p.add_argument('--train_ratio', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1)
    # Runtime
    p.add_argument('--outdir', default='./runs')
    p.add_argument('--run_name', default='resnet34_gaussian_noise_training')
    p.add_argument('--precision', default='32-true')
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--strategy', default='auto')
    return p


def main():
    args = build_argparser().parse_args()
    L.seed_everything(args.seed, workers=True)

    dm = RGBDataModuleFromTSV(
        rgb_train_root=args.rgb_train_root,
        rgb_val_root=args.rgb_val_root,
        paired_tsv_path=args.paired_tsv,
        ratio=args.train_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        noise_std=args.noise_std,
        noise_p=args.noise_p,
    )
    dm.setup()

    model = LitResNet34CENoise(
        lr=args.lr, weight_decay=args.weight_decay, ckpt_path=args.pretrained_ckpt,
    )

    os.makedirs(args.outdir, exist_ok=True)
    logger = CSVLogger(save_dir=args.outdir, name=args.run_name)

    ckpt_cb = ModelCheckpoint(
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
        callbacks=[ckpt_cb, lrmon, epoch_cb],
        strategy=args.strategy,
        precision=args.precision,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    main()
