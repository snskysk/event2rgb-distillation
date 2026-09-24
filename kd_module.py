"""LightningModule for Event → RGB knowledge distillation (Part 2, `train_event_distilled.py`).

The teacher is the frozen DiST ResNet34 (2-channel event input); the student
is a fresh 3-channel ResNet34 trained on ImageNet-1k. Each step combines
a cross-entropy loss on a strongly-augmented RGB view with a KL loss on a
weakly-augmented RGB/event pair.

Setting ``alpha_ce = 0, alpha_kd = 1`` reproduces ``ED_SoftLabelOnly``,
while ``alpha_ce = 0.8, alpha_kd = 0.2`` reproduces ``ED_Student``.
"""

from __future__ import annotations

from typing import Callable, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from variant_utils import kd_kl_loss, make_warmup_cosine_lambda


class LitKDDiST(L.LightningModule):
    """CE (strong-aug RGB) + KL (DiST event teacher ↔ RGB student)."""

    def __init__(
        self,
        teacher: nn.Module,
        lr: float = 3e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        alpha_ce: float = 1.0,
        alpha_kd: float = 1.0,
        temperature: float = 1.0,
        train_mode: str = 'kd',
        mix_fn: Optional[Callable] = None,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = resnet34(weights=None, num_classes=1000)
        self.mix_fn = mix_fn
        self.ce_loss = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=['teacher', 'mix_fn'])

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        return self.student(x_rgb)

    def training_step(self, batch_dict, batch_idx):
        # --- CE branch (strong-aug RGB + optional MixUp/CutMix) ---
        batch_ce = batch_dict['ce']
        x_rgb_ce = batch_ce['rgb']
        y_ce = batch_ce['label'].long()

        if self.mix_fn is not None:
            x_rgb_ce, _ = self.mix_fn(x_rgb_ce, y_ce)
        logits_ce = self(x_rgb_ce)
        loss_ce = self.ce_loss(logits_ce, y_ce)
        acc_ce = (logits_ce.argmax(dim=1) == y_ce).float().mean()

        # --- KD branch (weak-aug RGB student ↔ event teacher) ---
        batch_kd = batch_dict['kd']
        x_rgb_kd = batch_kd['rgb']
        x_event_kd = batch_kd['event']
        y_kd = batch_kd['label'].long()

        logits_s = self(x_rgb_kd)
        with torch.no_grad():
            logits_t = self.teacher(x_event_kd)

        loss_kd = kd_kl_loss(logits_s, logits_t, self.hparams.temperature)
        acc_kd = (logits_s.argmax(dim=1) == y_kd).float().mean()

        total_loss = self.hparams.alpha_ce * loss_ce + self.hparams.alpha_kd * loss_kd

        self.log_dict({
            'train/loss_total': total_loss,
            'train/loss_ce': loss_ce,
            'train/loss_kd': loss_kd,
            'train/acc_ce': acc_ce,
            'train/acc_kd': acc_kd,
        }, prog_bar=True, on_epoch=True, on_step=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x_rgb, y = batch
        else:
            x_rgb = batch['rgb']
            y = batch['label'].long()

        logits = self(x_rgb)
        loss = self.ce_loss(logits, y)
        acc1 = (logits.argmax(dim=1) == y).float().mean()
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/acc1', acc1, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            make_warmup_cosine_lambda(self.hparams.warmup_epochs, self.trainer.max_epochs),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val/acc1',
            },
        }

    def on_fit_start(self):
        self.teacher.to(self.device).eval()
