import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from base.data.data_container import DataContainer
from base.models.model_container import ModelContainer
from base.train.common_trainer import CommonTrainer
from base.train.metrics import accuracy


OPTIMIZERS = {
    'SGD': lambda params, cfg: optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay),
    'Adam': lambda params, cfg: optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay),
    'AdamW': lambda params, cfg: optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay),
}


class CNNTrainer(CommonTrainer):
    def __init__(self, cfg, model_container: ModelContainer, data_container: DataContainer, **kwargs):
        super(CNNTrainer, self).__init__(cfg, model_container, data_container)
        print(f'Initializing trainer {self.__class__.__name__}...')
        self.init_env()
        self.init_optimizer()
        self.init_scheduler()
        self.prep_train()
        self.loss_func = nn.CrossEntropyLoss()
        self.debug = getattr(self.cfg, 'debug', False)
        self.debug_input = getattr(self.cfg, 'debug_input', False)
        self.debug_labels = getattr(self.cfg, 'debug_labels', False)

    def init_optimizer(self, **kwargs):
        """Initialize self.optimizer using self.cfg."""
        model = self.model_container.models['model']
        opt_type = getattr(self.cfg, 'optimizer', 'SGD')
        freeze = getattr(self.cfg, 'freeze', False) or getattr(self.cfg, 'train_classifier', False)

        params = filter(lambda p: p.requires_grad, model.parameters()) if freeze else model.parameters()
        if freeze:
            print('Freezing weights!')

        builder = OPTIMIZERS.get(opt_type)
        if builder is None:
            print(f'Unknown optimizer {opt_type}, falling back to SGD')
            builder = OPTIMIZERS['SGD']
        else:
            print(f'Using {opt_type} as optimizer')
        self.optimizer = builder(params, self.cfg)

    @staticmethod
    def mixup_data(x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    @staticmethod
    def cutmix_data(x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(x.size(0)).to(x.device)

        H, W = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        return mixed_x, y, y[index], lam

    def _mixed_loss(self, input_data, label, mix_fn):
        mixed_x, targets_a, targets_b, lam = mix_fn(input_data, label)
        pred = self.model_container.infer('model', mixed_x, False)
        loss = lam * self.loss_func(pred, targets_a) + (1 - lam) * self.loss_func(pred, targets_b)
        return pred, loss

    def train(self, data_dict, **kwargs):
        input_data = data_dict['input_data']
        label = data_dict['label']

        self.model_container.set_train(['model'])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(self.devices[0])

        self.optimizer.zero_grad()

        apply_cutmix = getattr(self.cfg, 'cutmix', False)
        apply_mixup = getattr(self.cfg, 'mixup', False)

        # If both CutMix and MixUp are enabled, pick one at random per batch.
        if apply_cutmix and apply_mixup:
            mix_fn = self.cutmix_data if np.random.rand() < 0.5 else self.mixup_data
            pred, loss = self._mixed_loss(input_data, label, mix_fn)
        elif apply_cutmix:
            pred, loss = self._mixed_loss(input_data, label, self.cutmix_data)
        elif apply_mixup:
            pred, loss = self._mixed_loss(input_data, label, self.mixup_data)
        else:
            pred = self.model_container.infer('model', input_data, False)
            loss = self.loss_func(pred, label)

        acc_1, acc_5 = accuracy(pred.cpu(), label.cpu(), topk=(1, min(5, pred.shape[-1])))
        loss.backward()
        self.optimizer.step()

        if self.debug:
            if self.debug_input:
                self.inspect_input(input_data)
            if self.debug_labels:
                self.inspect_labels(pred, label, acc_1)

        return loss.item(), acc_1, acc_5

    def test(self, data_dict, **kwargs):
        input_data = data_dict['input_data']
        label = data_dict['label']

        self.model_container.set_eval(['model'])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(self.devices[0])

        pred = self.model_container.infer('model', input_data, True)
        tot_num = len(pred)
        acc_1, acc_5 = accuracy(pred.cpu(), label.cpu(), topk=(1, min(5, pred.shape[-1])))
        num_correct = int(acc_1 * tot_num)

        if self.debug:
            if self.debug_input:
                self.inspect_input(input_data)
            if self.debug_labels:
                self.inspect_labels(pred, label, acc_1)

        return acc_1, acc_5, num_correct, tot_num, pred

    def save_model(self, total_epoch, total_iter, total_val_iter, **kwargs):
        """Save model on self.exp_save_dir."""
        save_mode = self.cfg.save_by
        if save_mode == 'iter':
            counter, key = total_iter, 'iter'
        elif save_mode == 'epoch':
            counter, key = total_epoch, 'epoch'
        else:
            return
        if counter % self.cfg.save_every != 0:
            return

        model = self.model_container.models['model']
        multi_gpu = torch.cuda.device_count() >= 1 and self.cfg.parallel
        state_dict = model.module.state_dict() if multi_gpu else model.state_dict()

        save_dir = self.exp_save_dir / 'model_log' / f'checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar'
        torch.save({key: counter, 'state_dict': state_dict}, save_dir)

    def prep_train(self):
        """Auxiliary function for initialization (data parallelism, loading pretrained weights, etc.)."""
        self.model_container.load_saved()
        self.model_container.parallelize(['model'], self.devices)
        self.model_container.print_model_size(['model'])

    def inspect_labels(self, pred, label, acc_1):
        """Debugging utility for inspecting ground-truth / predicted labels."""
        thres = getattr(self.cfg, 'acc_threshold', 0.1)
        if acc_1 < thres:
            unq, cnt = torch.unique(pred.argmax(-1), return_counts=True)
            gt = self.data_container.dataset[self.cfg.mode].labels[label[0]]
            top = self.data_container.dataset[self.cfg.mode].labels[unq[cnt.argmax()]]
            print(f'GT label: {gt} {label[0]}')
            print(f'Most predicted: {top} {unq[cnt.argmax()]}')

    def inspect_input(self, input_data):
        """Debugging utility for visualizing input data."""
        inspect_channel = getattr(self.cfg, 'inspect_channel', 0)
        inspect_index = getattr(self.cfg, 'inspect_index', 0)

        if inspect_index == 'random':
            inspect_index = random.randint(0, len(input_data) - 1)

        if isinstance(inspect_channel, int):
            tmp = input_data[inspect_index].permute(1, 2, 0)[:, :, inspect_channel]
            plt.imshow(255 * tmp.cpu().numpy())
            plt.show()
        elif inspect_channel == 'all':
            fig = plt.figure(figsize=(50, 50))
            tmp = input_data[inspect_index].permute(1, 2, 0)
            num_channel = tmp.shape[-1]
            for i in range(num_channel):
                fig.add_subplot(num_channel // 2, 2, i + 1)
                plt.imshow(255 * tmp[..., i].cpu().numpy())
            plt.show()
