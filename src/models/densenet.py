import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pretrainedmodels

from .head import Head
from .pooling import GeneralizedMeanPool2d
from .models_lpf import densenet as lpf_densenet


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class DenseNet(nn.Module):
    def __init__(self,
                 conf,
                 arch_name='densenet121',
                 pretrained=True,
                 input_channel=3,                 
                 input_size=224):
        super(DenseNet, self).__init__()
        if "lpf" in arch_name:
            arch_name = arch_name.strip("lpf_")
            self.base_model = lpf_densenet.__dict__[arch_name](filter_size=3)
            if pretrained:
                self.base_model.load_state_dict(
                    torch.load(f'/home/korei/workspace/antialiased-cnns/weights/{arch_name}_lpf3.pth.tar')['state_dict'])
        else:
            self.base_model = torchvision.models.__dict__[arch_name](pretrained="imagenet")
        
        if isinstance(input_size, tuple):
            ksize = (input_size[0] // 32, input_size[1] // 32)
        else:
            ksize = input_size // 32

        self.base_model.features.conv0 = nn.Conv2d(input_channel, 64,
                                                   kernel_size=7, stride=2, padding=3,
                                                   bias=False)
        self.pooling = GeM()

        self.dim_feats = self.base_model.classifier.in_features  # = 1024
        self.fc_gr = Head(self.dim_feats, conf.gr_size)
        self.fc_vd = Head(self.dim_feats, conf.vd_size)
        self.fc_cd = Head(self.dim_feats, conf.cd_size)            

        self.out_size = ksize

    def forward(self, data):
        x = self.base_model.features(data)
        x = self.pooling(x)
        x = x.view(-1, self.dim_feats)

        gr = self.fc_gr(x)
        vd = self.fc_vd(x)
        cd = self.fc_cd(x)

        return np.array([gr, vd, cd])
