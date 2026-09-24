import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    squeezenet1_1,
    vgg16_bn,
    convnext_base,
)

from base.models.model_container import ModelContainer
from train_event_teacher.models.replknet import create_RepLKNet31B


MODEL_BUILDERS = {
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'ResNet101': resnet101,
    'ResNet152': resnet152,
    'SqueezeNet1_1': squeezenet1_1,
    'VGG16': vgg16_bn,
    'ConvNeXt_B': convnext_base,
}


class CNNContainer(ModelContainer):
    def __init__(self, cfg, **kwargs):
        super(CNNContainer, self).__init__(cfg)
        print(f'Initializing model container {self.__class__.__name__}...')
        self.gen_model()

    def gen_model(self):
        """Generate models for self.models."""
        use_pretrained = getattr(self.cfg, 'pretrained', False)
        num_classes = getattr(self.cfg, 'num_classes', 1000)
        pretrained_num_classes = getattr(self.cfg, 'pretrained_num_classes', 1000)
        model_name = self.cfg.model

        if model_name in MODEL_BUILDERS:
            model = MODEL_BUILDERS[model_name](pretrained=use_pretrained, num_classes=pretrained_num_classes)
        elif model_name == 'RepLKNet31B':
            # Disable gradient checkpointing so it does not conflict with DataParallel.
            model = create_RepLKNet31B(num_classes=pretrained_num_classes, use_checkpoint=False)
        else:
            raise AttributeError('Invalid model name')

        if num_classes != pretrained_num_classes:
            model.fc = nn.Linear(512, num_classes)

        channels = getattr(self.cfg, 'channel_size', 4)
        kernel_size = getattr(self.cfg, 'kernel_size', 7)
        self._adapt_input_channels(model, model_name, channels, kernel_size)

        if getattr(self.cfg, 'freeze', False):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.conv1.parameters():
                param.requires_grad = True
            for param in model.layer1.parameters():
                param.requires_grad = True

        if getattr(self.cfg, 'train_classifier', False):
            print('Training only last layer!')
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

        print(f'Using {model_name} for training')
        self.models['model'] = model

    @staticmethod
    def _adapt_input_channels(model, model_name, channels, kernel_size):
        """Replace the first convolution so the model accepts `channels` input channels."""
        if 'ResNet' in model_name:
            model.conv1 = nn.Conv2d(channels, 64, kernel_size=kernel_size, stride=2, padding=3, bias=False)
        elif model_name == 'SqueezeNet1_1':
            model.features[0] = nn.Conv2d(channels, 64, kernel_size=7, stride=2)
        elif model_name == 'VGG16':
            model.features[0] = nn.Conv2d(channels, 64, kernel_size=kernel_size, stride=1, padding=1)
        elif model_name == 'ConvNeXt_B':
            model.features[0][0] = nn.Conv2d(channels, 128, kernel_size=4, stride=4)
        elif model_name == 'RepLKNet31B':
            # model.stem[0] is a conv_bn_relu Sequential whose entry 'conv' is the input conv layer.
            original_conv = model.stem[0].conv
            model.stem[0].conv = nn.Conv2d(
                in_channels=channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=(original_conv.bias is not None),
            )

    def load_saved(self):
        pretrained_num_classes = getattr(self.cfg, 'pretrained_num_classes', None)
        if pretrained_num_classes is not None:
            self.models['model'].fc = nn.Linear(512, pretrained_num_classes)

        if self.cfg.load_model is not None:
            print(f'Loading model from {self.cfg.load_model}')
            state = torch.load(self.cfg.load_model)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if isinstance(state, dict):
                state = {k.replace('module.', ''): v for k, v in state.items()}
            missing, unexpected = self.models['model'].load_state_dict(state, strict=False)
            print(f'[Teacher load] missing={len(missing)}, unexpected={len(unexpected)}')

        if self.cfg.mode != 'test':
            num_classes = getattr(self.cfg, 'num_classes', 1000)
            if not getattr(self.cfg, 'keep_fc', True):
                self.models['model'].fc = nn.Linear(512, num_classes)
