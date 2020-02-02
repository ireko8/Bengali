import numpy as np
import torch
import torch.nn as nn
from .attention import Decoder, SCModule, SEModule
from .resnet import ResNet
from .head import Head
from .pooling import GeM


class UpsampleAdd(nn.Module):

    def __init__(self, out_channels, scale_factor=2):
        super(UpsampleAdd, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor,
                                    mode='bilinear',
                                    align_corners=True)
        self.sc = SCModule(out_channels)

    def forward(self, x, y):
        x = self.upsample(x)  # + y
        s = self.sc(x)
        return s + x


class UpsampleConcat(nn.Module):

    def __init__(self, out_channels, scale_factor=2):
        super(UpsampleAdd, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor,
                                    mode='bilinear',
                                    align_corners=True)
        self.sc = SCModule(out_channels)
        self.se = SEModule(out_channels)

    def forward(self, x, y):
        x = self.upsample(x) + y
        g_sc = self.sc(x)
        g_se = self.se(x)
        s =  g_sc * x + g_se * x
        return s



class FeaturePyramidNet(nn.Module):
    def __init__(self,
                 base_model):
        super(FeaturePyramidNet, self).__init__()
        self.base_model = base_model

        self.fsize = base_model.dim_feats
        self.out_size = base_model.out_size

        self.center = nn.Conv2d(self.fsize, 256, kernel_size=1)

        self.upa1 = UpsampleAdd(256)
        self.upa2 = UpsampleAdd(256)
        self.upa3 = UpsampleAdd(256)

        self.lat3 = nn.Conv2d(1024, 256, 1)
        self.lat2 = nn.Conv2d(512, 256, 1)
        self.lat1 = nn.Conv2d(256, 256, 1)

        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, 3, padding=1)

        self.avg4 = nn.AvgPool2d(self.out_size)
        self.avg3 = nn.AvgPool2d(self.out_size * 2)
        self.avg2 = nn.AvgPool2d(self.out_size * 4)
        self.avg1 = nn.AvgPool2d(self.out_size * 8)

        self.out4 = Head(self.fsize)
        self.out3 = Head(256)
        self.out2 = Head(256)
        self.out1 = Head(256)

    def forward(self, x):
        # x = self.base_model.base_model.layer0(x)
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        l1 = self.base_model.base_model.layer1(x)
        l2 = self.base_model.base_model.layer2(l1)
        l3 = self.base_model.base_model.layer3(l2)
        l4 = self.base_model.base_model.layer4(l3)
        p4 = self.avg4(l4).view(-1, 2048)
        p4 = self.out4(p4)

        s4 = self.center(l4)

        f3 = self.lat3(l3)
        f3 = self.upa3(s4, f3)
        # s3 = self.smooth3(f3)
        p3 = self.avg3(f3).view(-1, 256)
        p3 = self.out3(p3)

        f2 = self.lat2(l2)
        f2 = self.upa2(f3, f2)
        # s2 = self.smooth2(f2)
        p2 = self.avg2(f2).view(-1, 256)
        p2 = self.out2(p2)

        f1 = self.lat1(l1)
        f1 = self.upa1(f2, f1)
        # s1 = self.smooth1(f1)
        p1 = self.avg1(f1).view(-1, 256)
        p1 = self.out1(p1)

        # p = torch.cat([p1, p2, p3, p4], dim=1)
        # pred = self.out(p)
        return [p1, p2, p3, p4]


class FeaturePyramidNetV2(nn.Module):
    def __init__(self,
                 base_model):
        super(FeaturePyramidNetV2, self).__init__()
        self.base_model = base_model

        self.fsize = base_model.dim_feats
        self.out_size = base_model.out_size

        self.center = nn.Conv2d(self.fsize, 256, kernel_size=1)

        self.dc1 = Decoder(256 + 1024, 256, 256, scse=True)
        self.dc2 = Decoder(256 + 512, 256, 256, scse=True)
        self.dc3 = Decoder(256 + 256, 256, 256, scse=True)

        self.avg4 = nn.AvgPool2d(self.out_size)
        self.avg3 = nn.AvgPool2d(self.out_size * 2)
        self.avg2 = nn.AvgPool2d(self.out_size * 4)
        self.avg1 = nn.AvgPool2d(self.out_size * 8)

        self.out4 = Head(256)
        self.out3 = Head(256)
        self.out2 = Head(256)
        self.out1 = Head(256)

    def forward(self, x):
        x = self.base_model.base_model.layer0(x)
        l1 = self.base_model.base_model.layer1(x)
        l2 = self.base_model.base_model.layer2(l1)
        l3 = self.base_model.base_model.layer3(l2)
        l4 = self.base_model.base_model.layer4(l3)

        s4 = self.center(l4)
        p4 = self.avg4(s4).view(-1, 256)
        p4 = self.out4(p4)

        f3 = self.dc1(s4, l3)
        p3 = self.avg3(f3).view(-1, 256)
        p3 = self.out3(p3)

        f2 = self.dc2(f3, l2)
        p2 = self.avg2(f2).view(-1, 256)
        p2 = self.out2(p2)

        f1 = self.dc3(f2, l1)
        p1 = self.avg1(f1).view(-1, 256)
        p1 = self.out1(p1)

        return [p1, p2, p3, p4]


class FeaturePyramidNetV3(nn.Module):
    def __init__(self,
                 conf,
                 base_model):
        super(FeaturePyramidNetV3, self).__init__()
        self.base_model = base_model

        self.fsize = base_model.dim_feats
        self.out_size = base_model.out_size

        self.center = nn.Conv2d(self.fsize, 64, kernel_size=1)

        self.upa1 = UpsampleAdd(64)
        self.upa2 = UpsampleAdd(64)
        self.upa3 = UpsampleAdd(64)

        self.lat3 = nn.Conv2d(1024, 64, 1)
        self.lat2 = nn.Conv2d(512, 64, 1)
        self.lat1 = nn.Conv2d(256, 64, 1)

        self.smooth3 = nn.Conv2d(64, 64, 3, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, 3, padding=1)
        self.smooth1 = nn.Conv2d(64, 64, 3, padding=1)

        self.avg4 = GeM()
        self.avg3 = GeM()
        self.avg2 = GeM()
        self.avg1 = GeM()

        self.fc_gr = Head(256, conf.gr_size)
        self.fc_vd = Head(256, conf.vd_size)
        self.fc_cd = Head(256, conf.cd_size)

    def forward(self, x):
        x = self.base_model.base_model.layer0(x)

        l1 = self.base_model.base_model.layer1(x)
        l2 = self.base_model.base_model.layer2(l1)
        l3 = self.base_model.base_model.layer3(l2)
        l4 = self.base_model.base_model.layer4(l3)

        s4 = self.center(l4)
        p4 = self.avg4(s4).view(-1, 64)

        f3 = self.lat3(l3)
        f3 = self.upa3(s4, f3)
        # s3 = self.smooth3(f3)
        p3 = self.avg3(f3).view(-1, 64)

        f2 = self.lat2(l2)
        f2 = self.upa2(f3, f2)
        # s2 = self.smooth2(f2)
        p2 = self.avg2(f2).view(-1, 64)

        f1 = self.lat1(l1)
        f1 = self.upa1(f2, f1)
        # s1 = self.smooth1(f1)
        p1 = self.avg1(f1).view(-1, 64)

        x = torch.cat([p1, p2, p3, p4], dim=1)
        gr = self.fc_gr(x)
        vd = self.fc_vd(x)
        cd = self.fc_cd(x)

        return np.array([gr, vd, cd]), f1


class FPResNet(nn.Module):

    def __init__(self, conf):
        super(FPResNet, self).__init__()
        base_model = ResNet(conf, arch_name=conf.arch,
                            input_size=conf.image_size)
        self.model = FeaturePyramidNetV3(conf, base_model)

    def forward(self, data):
        return self.model(data)
