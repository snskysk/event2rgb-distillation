"""Train a RGB ResNet34 student distilled from the DiST event teacher.

Paper variants produced by this script (pick via ``--alpha_ce`` / ``--alpha_kd``):
    - baseline          : alpha_ce=1.0, alpha_kd=0.0   (plain CE, no KD signal)
    - ED_Student        : alpha_ce=0.8, alpha_kd=0.2
    - ED_SoftLabelOnly  : alpha_ce=0.0, alpha_kd=1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from kd_datamodules import KDDataModule, SetEpochForSamplerCallback
from kd_module import LitKDDiST
from variant_utils import make_trainer


torch.set_float32_matmul_precision('high')


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Event → RGB knowledge distillation (ResNet34)')
    # Paths
    p.add_argument('--rgb_train_root', default='/datasets/imagenet/train')
    p.add_argument('--rgb_val_root', default='/datasets/imagenet/val')
    p.add_argument('--mapping_txt', default=str(CODE_ROOT / 'Datasets' / 'mapping.txt'))
    p.add_argument('--dist_weight', required=True,
                   help='Path to the DiST ResNet34 event-teacher checkpoint (see Part 1).')
    p.add_argument('--paired_tsv', default=str(CODE_ROOT / 'lists' / 'paired_train.tsv'))
    # KD hyperparameters (the main knobs for ablation)
    p.add_argument('--alpha_ce', type=float, required=True,
                   help='Weight of the cross-entropy loss (e.g. 0.0 for ED_SoftLabelOnly, 0.8 for ED_Student).')
    p.add_argument('--alpha_kd', type=float, required=True,
                   help='Weight of the KL distillation loss.')
    p.add_argument('--temperature', type=float, default=3.0)
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
    p.add_argument('--run_name', default='kd_resnet34')
    p.add_argument('--precision', default='32-true')
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--strategy', default='auto')
    return p


def main():
    args = build_argparser().parse_args()
    L.seed_everything(args.seed, workers=True)

    dm = KDDataModule(
        rgb_train_root=args.rgb_train_root,
        rgb_val_root=args.rgb_val_root,
        mapping_txt=args.mapping_txt,
        dist_weight_path=args.dist_weight,
        paired_tsv_path=args.paired_tsv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    dm.setup()

    model = LitKDDiST(
        teacher=dm.teacher,
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
    epoch_cb = SetEpochForSamplerCallback(lambda: dm._shared_sampler)

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
