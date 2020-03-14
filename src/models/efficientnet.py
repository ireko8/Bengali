import numpy as np
from torch import nn
import timm

from .head import Head
from .pooling import GeM


class EfficientNet(nn.Module):
    def __init__(self,
                 conf,
                 arch_name='efficientnet-b3',
                 input_channel=1,
                 pretrained=True):
        super(EfficientNet, self).__init__()

        self.base_model = timm.create_model(
            arch_name, pretrained=True)
        self.base_model.conv_stem = nn.Conv2d(input_channel, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.out_size = self.base_model.num_features
        self.pooling = GeM()

        self.fc_gr = Head(self.out_size, conf.gr_size)
        self.fc_vd = Head(self.out_size, conf.vd_size)
        self.fc_cd = Head(self.out_size, conf.cd_size)

    def forward(self, data):

        x = self.base_model.forward_features(data)
        x = self.pooling(x).view(-1, self.out_size)

        gr = self.fc_gr(x)
        vd = self.fc_vd(x)
        cd = self.fc_cd(x)

        return np.array([gr, vd, cd])
