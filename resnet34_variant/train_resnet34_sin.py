"""SIN+IN co-training for ResNet-34 (Geirhos et al. 2019 style).

Trains a ResNet34 from scratch on ``ConcatDataset(ImageNet-1k train,
Stylized-ImageNet train)`` with a single shared WNID-keyed label space.
This is the comparison baseline used in the paper's "Event Distillation
vs. Existing Robustness Methods" discussion (Section 6.4): SIN imparts
shape bias by exposing the model to texture-randomised images, but is
not a knowledge-distillation method, so we provide a dedicated training
script rather than reusing ``train_event_distilled.py``.

Optimiser, schedule, and augmentation match ``teacher_train.py`` (the
Gray / B&W teacher trainer) so that downstream evaluations are
apples-to-apples with the rest of the Part 2 baselines.

Example usage (run from ``code_submittion/``):

    python resnet34_variant/train_resnet34_sin.py \
        --rgb_train_root  /datasets/imagenet/train \
        --rgb_val_root    /datasets/imagenet/val \
        --sin_train_root  /datasets/stylized_imagenet/train
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torchvision import datasets, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from variant_utils import (  # noqa: E402
    SetEpochForSamplerCallback,
    SyncedDistributedSampler,
    build_class_to_idx_from_train_root,
    make_trainer,
    make_warmup_cosine_lambda,
    tfm_rgb_strong,
    tfm_rgb_weak,
)


CODE_ROOT = Path(__file__).resolve().parent.parent
torch.set_float32_matmul_precision('high')


class ImageFolderWithSharedIdx(Dataset):
    """``ImageFolder``-style dataset that uses an externally supplied ``class_to_idx``.

    Used so that the IN and SIN sub-datasets share the same label space
    (a WNID present in only one of the two trees still gets the canonical
    index built from the IN train root).
    """

    def __init__(self, root: str, class_to_idx: Dict[str, int], tfm):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.tfm = tfm
        self.samples = []
        for wnid, cls_idx in class_to_idx.items():
            cls_dir = self.root / wnid
            if not cls_dir.is_dir():
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.samples.append((str(cls_dir / fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert('RGB')
            x = self.tfm(im)
        return x, torch.tensor(label, dtype=torch.long)


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


class SinInDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self._sampler: Optional[Sampler[int]] = None

    def setup(self, stage=None):
        args = self.args
        class_to_idx = build_class_to_idx_from_train_root(args.rgb_train_root)

        tfm_train = tfm_rgb_strong()
        in_ds = ImageFolderWithSharedIdx(args.rgb_train_root, class_to_idx, tfm_train)
        sin_ds = ImageFolderWithSharedIdx(args.sin_train_root, class_to_idx, tfm_train)
        self.train_ds = ConcatDataset([in_ds, sin_ds])
        print(f'[data] IN={len(in_ds)} SIN={len(sin_ds)} concat={len(self.train_ds)}')

        self.val_ds = datasets.ImageFolder(args.rgb_val_root, transform=tfm_rgb_weak())
        assert self.val_ds.class_to_idx == class_to_idx, 'Val class_to_idx mismatch'

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
    p = argparse.ArgumentParser(description='Train a ResNet34 with SIN+IN co-training (Geirhos et al. 2019)')
    # Data roots
    p.add_argument('--rgb_train_root', default='/datasets/imagenet/train',
                   help='Standard ImageNet-1k training tree (defines the canonical WNID -> class_idx map).')
    p.add_argument('--rgb_val_root', default='/datasets/imagenet/val',
                   help='Standard ImageNet-1k validation tree (used for evaluation).')
    p.add_argument('--sin_train_root', default='/datasets/stylized_imagenet/train',
                   help='Stylized-ImageNet training tree (parallel to rgb_train_root, same WNIDs).')
    # Training
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--seed', type=int, default=1)
    # Runtime
    p.add_argument('--outdir', default='./runs')
    p.add_argument('--run_name', default='resnet34_sin_in')
    p.add_argument('--precision', default='32-true')
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--strategy', default='auto')
    return p


def main():
    args = build_argparser().parse_args()
    L.seed_everything(args.seed, workers=True)

    dm = SinInDataModule(args)
    dm.setup()

    model = LitResNet34CE(lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.outdir, exist_ok=True)
    logger = CSVLogger(save_dir=args.outdir, name=args.run_name)

    ckpt = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename='{epoch:02d}-{val_acc1:.4f}',
        monitor='val/acc1',
        mode='max',
        save_top_k=3,
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
        deterministic=False,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    print(f'[OK] best: {ckpt.best_model_path}')


if __name__ == '__main__':
    main()
