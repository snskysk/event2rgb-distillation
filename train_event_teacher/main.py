import faulthandler
faulthandler.enable()

import argparse
import configparser
import subprocess
import sys
from collections import namedtuple

sys.path.append('../')

from base.utils.parse_utils import parse_ini, parse_value
from train_event_teacher.data.data_container import ImageNetContainer
from train_event_teacher.models.model_container import CNNContainer
from train_event_teacher.train.trainer import CNNTrainer


def apply_override(cfg, override):
    """Parse a CLI override string and return an updated cfg namedtuple."""
    equality_split = override.split('=')
    num_equality = len(equality_split)
    assert num_equality > 0

    if num_equality == 2:
        override_dict = {equality_split[0]: parse_value(equality_split[1])}
    else:
        keys = [equality_split[0]]
        keys += [eq.split(',')[-1] for eq in equality_split[1:-1]]
        values = [eq.replace(',' + key, '') for eq, key in zip(equality_split[1:-1], keys[1:])]
        values.append(equality_split[-1])
        values = [v.replace('[', '').replace(']', '') for v in values]
        override_dict = {key: parse_value(value) for key, value in zip(keys, values)}

    cfg_dict = cfg._asdict()
    cfg_dict.update(override_dict)
    Config = namedtuple('Config', tuple(cfg_dict.keys()))
    cfg = Config(**cfg_dict)
    return cfg._replace(name=cfg.name + '_' + override)


def main():
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument('--config', default='./config.ini', help='Config .ini file directory')
    parser.add_argument('--clean', action='store_true', help='Clean experiments')
    parser.add_argument('--background', action='store_true', help='Run experiment in the background')
    parser.add_argument('--override', default=None, help='Arguments for overriding config')
    parser.add_argument('--cutmix', action='store_true', help='Enable CutMix')
    parser.add_argument('--mixup', action='store_true', help='Enable MixUp')
    args = parser.parse_args()

    cfg = parse_ini(args.config)

    print('Display Config:')
    with open(args.config, 'r') as f:
        print(f.read())

    if args.override is not None:
        cfg = apply_override(cfg, args.override)

    cfg_dict = cfg._asdict()
    cfg_dict['cutmix'] = args.cutmix
    cfg_dict['mixup'] = args.mixup
    Config = namedtuple('Config', tuple(cfg_dict.keys()))
    cfg = Config(**cfg_dict)

    data_container = ImageNetContainer(cfg)
    model_container = CNNContainer(cfg)
    trainer = CNNTrainer(cfg, model_container, data_container)

    config = configparser.ConfigParser()
    config.add_section('Default')
    for key, value in cfg._asdict().items():
        if key != 'name':
            config['Default'][key] = str(value).replace('[', '').replace(']', '')
        else:
            config['Default'][key] = str(value)

    with open(trainer.exp_save_dir / 'config.ini', 'w') as configfile:
        config.write(configfile)

    print(model_container.models['model'])

    if args.clean:
        print('Cleaning experiments!')
        subprocess.call(['rm', '-rf', trainer.exp_save_dir])
        exit()

    trainer.run()


if __name__ == '__main__':
    main()
