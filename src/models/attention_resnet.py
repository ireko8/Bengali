import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


class Conv2dBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 padding=0):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SCModule(nn.Module):
    def __init__(self, channels):
        super(SCModule, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels):
        super(SEModule, self).__init__()
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 2)
        self.fc2 = nn.Linear(channels // 2, channels)

    def forward(self, x):
        x = self.gap(x).view(-1, self.channels)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.sigmoid(self.fc2(x))
        return x.view(-1, self.channels, 1, 1)


class Decoder(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, scse=False):
        super(Decoder, self).__init__()
        self.conv1 = Conv2dBN(in_channels, inter_channels, 3, padding=1)
        self.conv2 = Conv2dBN(inter_channels, out_channels, 3, padding=1)
        self.scse = scse
        if scse:
            self.sc = SCModule(out_channels)
            self.se = SEModule(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], dim=1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        if self.scse:
            g_sc = self.sc(x)
            g_se = self.se(x)
            x = g_sc * x + g_se * x
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, scse=False):
        super(SimpleDecoder, self).__init__()
        self.conv = Conv2dBN(in_channels, out_channels, 3, padding=1)

    def forward(self, x, scale_factor):
        x = F.relu(self.conv(x), inplace=True)
        x = F.upsample(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        return x


class SimpleAttentionNet(nn.Module):
    def __init__(self,
                 arch_name='se_resnet50',
                 input_channel=3,
                 input_size=224,
                 num_classes=28):
        super(SimpleAttentionNet, self).__init__()
        base_model = pretrainedmodels.__dict__[arch_name](pretrained='imagenet')

        self.pointwise_layer = nn.Sequential(
            nn.Conv2d(input_channel, 3, kernel_size=(1, 1)), nn.BatchNorm2d(3),
            nn.ReLU())

        ksize = input_size // 32
        if input_size != 224:
            base_model.layer0.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            base_model.avg_pool = nn.AvgPool2d(kernel_size=ksize, stride=1)

        self.encoder = nn.Sequential(
            # self.pointwise_layer,
            base_model.layer0,
            base_model.layer1,  # 32
            base_model.layer2,  # 128
            base_model.layer3,
            base_model.layer4)  # 512

        self.avg_pool = base_model.avg_pool
        self.head = nn.Linear(2048, num_classes)

        self.center = nn.Sequential(
            Conv2dBN(2048, 512, 3, padding=1), nn.ReLU(),
            Conv2dBN(512, 256, 3, padding=1), nn.ReLU())

        self.decoder = nn.Sequential(Decoder(256, 128, 128),
                                     Decoder(128, 128, 64),
                                     Decoder(64, 64, 32),
                                     nn.Conv2d(32, 32, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(32, 28, 1, padding=0),
                                     nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        cent = self.center(x)
        dec = self.decoder(cent)
        avg = self.avg_pool(x)
        clf = self.head(avg.view(-1, 2048))
        return clf, dec


class SimpleAttentionDenseNet(nn.Module):
    def __init__(self,
                 arch='densenet121',
                 input_channel=3,
                 input_size=224,
                 num_classes=28):
        super(SimpleAttentionDenseNet, self).__init__()
        base_model = pretrainedmodels.__dict__[arch](pretrained='imagenet')

        ksize = input_size // 4
        if input_size != 224:
            base_model.features.conv0 = nn.Conv2d(3, 64, 7, stride=2, padding=3)

        self.encoder = base_model.features

        self.avg_pool = nn.AvgPool2d(ksize)
        self.head = nn.Linear(num_classes, num_classes)

        self.center = nn.Sequential(
            Conv2dBN(1024, 512, 3, padding=1), nn.ReLU(),
            Conv2dBN(512, 256, 3, padding=1), nn.ReLU())

        self.decoder = nn.Sequential(Decoder(256, 128, 128),
                                     Decoder(128, 128, 64),
                                     Decoder(64, 64, 32),
                                     nn.Conv2d(32, 32, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(32, 28, 1, padding=0))
        self.dec_out = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        cent = self.center(x)
        dec = self.decoder(cent)
        avg = self.avg_pool(dec)
        dec = self.dec_out(dec)
        clf = self.head(avg.view(-1, 28))
        return clf, dec


class Decoder2(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, scse=False):
        super(Decoder2, self).__init__()
        self.conv1 = Conv2dBN(in_channels, inter_channels, 3, padding=1)
        self.conv2 = Conv2dBN(inter_channels, out_channels, 3, padding=1)
        self.scse = scse
        if scse:
            self.sc = SCModule(out_channels)
            self.se = SEModule(out_channels)

    def forward(self, x, e=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e:
            x = torch.cat([x, e], dim=1)

        if self.scse:
            g_sc = self.sc(x)
            g_se = self.se(x)
            x = g_sc * x + g_se * x
        return x


class EncoderDecoder(nn.Module):
    def __init__(self,
                 base_model,
                 num_classes=28):
        super(EncoderDecoder, self).__init__()
        self.base_model = base_model

        self.decoder = nn.Sequential(Decoder2(2048, 512, 512),  # 8   -> 16
                                     Decoder2(512, 128, 128),   # 16  -> 32
                                     Decoder2(128, 32, 32),     # 32  -> 64
                                     Decoder2(32, 8, 8),        # 64  -> 128
                                     # Decoder2(8, 4, 4),         # 128 -> 256
                                     nn.Conv2d(8, 4, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(4, 3, 1, padding=0),
                                     nn.Sigmoid())

    def forward(self, x):
        x = self.base_model.pointwise_layer(x)
        x = self.base_model.base_model.layer0(x)
        x = self.base_model.base_model.layer1(x)
        x = self.base_model.base_model.layer2(x)
        x = self.base_model.base_model.layer3(x)
        x = self.base_model.base_model.layer4(x)

        dec = self.decoder(x)
        return dec


class AttentionResNet(nn.Module):
    def __init__(self, conf, base_model):                 
        super(AttentionResNet, self).__init__()

        self.center = nn.Sequential(
            Conv2dBN(2048, 512, 3, padding=1), nn.ReLU(inplace=True),
            Conv2dBN(512, 256, 3, padding=1), nn.ReLU(inplace=True))

        self.decoder4 = SimpleDecoder(2048, 64)
        self.decoder3 = SimpleDecoder(1024, 64)
        self.decoder2 = SimpleDecoder(512, 64)
        self.decoder1 = SimpleDecoder(256, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, padding=0), nn.Sigmoid())

        self.pooling = base_model.pooling
        self.resnet = base_model

    def forward(self, x):
        x = self.resnet.base_model.layer0(x)

        e1 = self.resnet.base_model.layer1(x)
        e2 = self.resnet.base_model.layer2(e1)
        e3 = self.resnet.base_model.layer3(e2)
        e4 = self.resnet.base_model.layer4(e3)

        x = self.resnet.pooling(e4).view(-1, self.resnet.dim_feats)

        gr = self.resnet.fc_gr(x)
        vd = self.resnet.fc_vd(x)
        cd = self.resnet.fc_cd(x)

        h = torch.cat((self.decoder4(e4, scale_factor=4),
                       self.decoder3(e3, scale_factor=2),
                       self.decoder2(e2, scale_factor=1)), dim=1)

        logit = self.logit(h)
        return np.array([gr, vd, cd]), logit
