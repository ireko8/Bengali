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

