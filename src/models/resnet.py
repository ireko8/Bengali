import numpy as np
import torch
from torch import nn
import torchvision
import pretrainedmodels

from .head import Head
from .pooling import GeM
from .models_lpf import resnet as lpf_resnet
    

class ResNet(nn.Module):
    def __init__(self,
                 conf,
                 arch_name='resnet18',
                 input_channel=3,
                 input_size=224,
                 pretrained=False):
        super(ResNet, self).__init__()
        self.se = False
        
        if "se" in arch_name:
            self.base_model = pretrainedmodels.__dict__[arch_name](pretrained="imagenet")
            self.se = True
        elif "lpf" in arch_name:
            arch_name = arch_name.strip("lpf_")
            self.base_model = lpf_resnet.__dict__[arch_name](filter_size=3)
            if pretrained:
                self.base_model.load_state_dict(
                    torch.load(f'/home/korei/workspace/antialiased-cnns/weights/{arch_name}_lpf3.pth.tar')['state_dict'])
        else:
            self.base_model = torchvision.models.__dict__[arch_name](pretrained="imagenet")

        if isinstance(input_size, tuple):
            ksize = (input_size[0] // 32, input_size[1] // 32)
        else:
            ksize = input_size // 32

        if self.se:
            self.base_model.layer0.conv1 = nn.Conv2d(input_channel, 64,
                                                     kernel_size=7, stride=2, padding=3,
                                                     bias=False)
        else:
            self.base_model.bn0 = nn.BatchNorm2d(input_channel)
            self.base_model.conv1 = nn.Conv2d(input_channel, 64,
                                              kernel_size=7, stride=2, padding=3,
                                              bias=False)

        # self.base_model.avgpool = nn.AvgPool2d(kernel_size=ksize)
        # self.base_model.maxpool = nn.MaxPool2d(kernel_size=ksize)
        self.pooling = GeM()

        if self.se:
            self.dim_feats = self.base_model.last_linear.in_features  # = 2048
        else:
            self.dim_feats = self.base_model.fc.in_features  # = 2048

        head_input_size = self.dim_feats
        self.fc_gr = Head(head_input_size, conf.gr_size)
        self.fc_vd = Head(head_input_size, conf.vd_size)
        self.fc_cd = Head(head_input_size, conf.cd_size)
        self.out_size = ksize

    def forward(self, data):

        if self.se:
            x = self.base_model.layer0(data)
        else:
            x = self.base_model.conv1(data)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)

        l1 = self.base_model.layer1(x)
        l2 = self.base_model.layer2(l1)
        l3 = self.base_model.layer3(l2)
        l4 = self.base_model.layer4(l3)

        # avgpool = self.base_model.avgpool(l4).view(-1, self.dim_feats)
        # maxpool = self.base_model.maxpool(l4).view(-1, self.dim_feats)
        # x = torch.cat([avgpool, maxpool], dim=1)
        x = self.pooling(l4).view(-1, self.dim_feats)

        gr = self.fc_gr(x)
        vd = self.fc_vd(x)
        cd = self.fc_cd(x)

        return np.array([gr, vd, cd])
