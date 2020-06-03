import torch.nn as nn
import pretrainedmodels
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import get_root_logger

from ..builder import BACKBONES
@BACKBONES.register_module()
class SeResNet(nn.Module):

    def __init__(self):
        super(SeResNet, self).__init__()
        backbone = pretrainedmodels.__dict__["se_resnet50"](num_classes=1000, pretrained=None)
        self.layer0 = backbone.layer0
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        return [x1,x2,x3,x4]
    """
    ([b, 64, 125, 150])
    ([b, 256, 125, 150])
    ([b, 512, 63, 75])
    ([b, 1024, 32, 38])
    """
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
