import torch
from torch.utils.data import DataLoader

from base.data.data_container import DataContainer
from train_event_teacher.data.imagenet import ImageNetDataset


def dict_collate_fn(list_data):
    """Collate a list of (event_tensor, label) pairs into a dict batch."""
    events, labels = list(zip(*list_data))
    return {
        'input_data': torch.stack(events, dim=0),
        'label': torch.LongTensor(labels),
    }


class ImageNetContainer(DataContainer):
    def __init__(self, cfg, **kwargs):
        super(ImageNetContainer, self).__init__(cfg)
        print(f'Initializing data container {self.__class__.__name__}...')
        self.gen_dataset()
        self.gen_dataloader()

    def gen_dataset(self, **kwargs):
        dataset_name = getattr(self.cfg, 'dataset_name', 'imagenet')
        if dataset_name != 'imagenet':
            return

        if self.cfg.mode == 'train':
            self.dataset['train'] = ImageNetDataset(self.cfg, mode='train')
            self.dataset['val'] = ImageNetDataset(self.cfg, mode='val')
        elif self.cfg.mode == 'test':
            self.dataset['test'] = ImageNetDataset(self.cfg, mode='test')
        else:
            raise AttributeError('Mode not provided')

    def gen_dataloader(self, **kwargs):
        assert self.dataloader is not None
        loader_kwargs = dict(
            collate_fn=dict_collate_fn,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=self.cfg.pin_memory,
        )

        if self.cfg.mode == 'train':
            self.dataloader['train'] = DataLoader(self.dataset['train'], shuffle=True, **loader_kwargs)
            self.dataloader['val'] = DataLoader(self.dataset['val'], shuffle=True, **loader_kwargs)
        elif self.cfg.mode == 'test':
            self.dataloader['test'] = DataLoader(self.dataset['test'], shuffle=False, **loader_kwargs)
        else:
            raise AttributeError('Mode not provided')
