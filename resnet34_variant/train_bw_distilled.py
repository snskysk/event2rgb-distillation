"""B&W-distillation baseline: binarised-image ResNet34 teacher -> RGB ResNet34 student.

Uses a single :class:`PairedRgbBwDataset` that returns the three views required
in one disk read: strong-aug RGB (CE), weak-aug RGB (student KD input), and
weak-aug B&W (teacher KD input). Reproduces the ``B&W Distilled`` row.
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

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from variant_utils import (
    PairedRgbBwDataset,
    SetEpochForSamplerCallback,
    SyncedDistributedSampler,
    build_class_to_idx_from_train_root,
    kd_kl_loss,
    load_pairs_tsv,
    load_resnet34_from_ckpt,
    make_trainer,
    make_warmup_cosine_lambda,
    subsample_paths_by_class_ratio,
    tfm_bw_weak,
    tfm_rgb_strong,
    tfm_rgb_weak,
)


torch.set_float32_matmul_precision('high')


class BwKDDataModule(L.LightningDataModule):
    def __init__(self, rgb_train_root, bw_train_root, rgb_val_root, paired_tsv_path,
                 ratio, seed, batch_size, num_workers):
        super().__init__()
        self.rgb_train_root = rgb_train_root
        self.bw_train_root = bw_train_root
        self.rgb_val_root = rgb_val_root
        self.paired_tsv_path = paired_tsv_path
        self.ratio = ratio
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.class_to_idx = None
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self._sampler: Optional[Sampler[int]] = None

    def setup(self, stage=None):
        self.class_to_idx = build_class_to_idx_from_train_root(self.rgb_train_root)
        rgb_paths, _, wnids, _ = load_pairs_tsv(self.paired_tsv_path)
        rgb_paths = subsample_paths_by_class_ratio(rgb_paths, wnids, ratio=self.ratio, seed=self.seed)

        self.train_ds = PairedRgbBwDataset(
            rgb_paths=rgb_paths,
            class_to_idx=self.class_to_idx,
            bw_root=self.bw_train_root,
            tfm_ce=tfm_rgb_strong(),
            tfm_kd=tfm_rgb_weak(),
            tfm_bw=tfm_bw_weak(),
        )
        self.val_ds = datasets.ImageFolder(self.rgb_val_root, transform=tfm_rgb_weak())
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


class LitKDBwTeacher(L.LightningModule):
    def __init__(self, teacher: nn.Module, lr: float, weight_decay: float,
                 alpha_ce: float, alpha_kd: float, temperature: float, warmup_epochs: int = 5):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = models.resnet34(weights=None, num_classes=1000)
        self.ce_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=['teacher'])

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        y = batch['label'].long()

        logits_ce = self(batch['rgb_ce'])
        loss_ce = self.ce_loss(logits_ce, y)
        acc_ce = (logits_ce.argmax(1) == y).float().mean()

        logits_s = self(batch['rgb_kd'])
        with torch.no_grad():
            logits_t = self.teacher(batch['bw_kd'])
        loss_kd = kd_kl_loss(logits_s, logits_t, self.hparams.temperature)
        acc_kd = (logits_s.argmax(1) == y).float().mean()

        total = self.hparams.alpha_ce * loss_ce + self.hparams.alpha_kd * loss_kd
        self.log_dict({
            'train/loss_total': total,
            'train/loss_ce': loss_ce,
            'train/loss_kd': loss_kd,
            'train/acc_ce': acc_ce,
            'train/acc_kd': acc_kd,
        }, prog_bar=True, on_epoch=True, on_step=False)
        return total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce_loss(logits, y)
        acc1 = (logits.argmax(1) == y).float().mean()
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/acc1', acc1, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, make_warmup_cosine_lambda(self.hparams.warmup_epochs, self.trainer.max_epochs),
        )
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}

    def on_fit_start(self):
        self.teacher.to(self.device).eval()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='B&W-distillation (B&W teacher -> RGB student)')
    p.add_argument('--rgb_train_root', default='/datasets/imagenet/train')
    p.add_argument('--rgb_val_root', default='/datasets/imagenet/val')
    p.add_argument('--bw_train_root', default='/datasets/imagenet_bw/train')
    p.add_argument('--paired_tsv', default=str(CODE_ROOT / 'lists' / 'paired_train.tsv'))
    p.add_argument('--teacher_ckpt', required=True,
                   help='Lightning checkpoint of the B&W ResNet34 teacher.')
    p.add_argument('--alpha_ce', type=float, default=0.8)
    p.add_argument('--alpha_kd', type=float, default=0.2)
    p.add_argument('--temperature', type=float, default=3.0)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=7.5e-4)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--train_ratio', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--outdir', default='./runs')
    p.add_argument('--run_name', default='kd_resnet34_bw_distill')
    p.add_argument('--precision', default='32-true')
    p.add_argument('--devices', type=int, nargs='+', default=[0])
    p.add_argument('--strategy', default='auto')
    return p


def main():
    args = build_argparser().parse_args()
    L.seed_everything(args.seed, workers=True)

    dm = BwKDDataModule(
        rgb_train_root=args.rgb_train_root,
        bw_train_root=args.bw_train_root,
        rgb_val_root=args.rgb_val_root,
        paired_tsv_path=args.paired_tsv,
        ratio=args.train_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    teacher = load_resnet34_from_ckpt(args.teacher_ckpt, freeze=True)
    print('[OK] teacher loaded')

    model = LitKDBwTeacher(
        teacher=teacher,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha_ce=args.alpha_ce,
        alpha_kd=args.alpha_kd,
        temperature=args.temperature,
    )

    os.makedirs(args.outdir, exist_ok=True)
    logger = CSVLogger(save_dir=args.outdir, name=args.run_name)

    ckpt = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename='{epoch:02d}-{val_acc1:.4f}',
        monitor='val/acc1',
        mode='max',
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
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    print(f'[OK] best: {ckpt.best_model_path}')


if __name__ == '__main__':
    main()
