import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import conf


class OnlineHardExampleMining(nn.Module):

    def __init__(self, loss, ratio=0.5):
        super(OnlineHardExampleMining, self).__init__()
        self.loss = loss
        self.min_size = int(conf.batch_size*ratio)

    def forward(self, x, y):
        size = min(self.min_size, x.size(0))
        loss, _ = torch.topk(self.loss(x, y), size)

        return loss.mean()


class LabelSmoothedCE(nn.Module):

    def __init__(self, lam=0.9):
        super(LabelSmoothedCE, self).__init__()
        self.lam = lam

    def forward(self, x, y):
        non_target = self.lam / x.size(1)
        
        one_hot = torch.zeros(x.size(), device=conf.device_name) + non_target
        one_hot.scatter_(1, y.view(-1, 1).long(), self.lam)

        logp = F.log_softmax(x, dim=1)
        loss = torch.sum(-logp * one_hot, dim=1)
        return loss.sum()
        

class RecallLoss(nn.Module):

    def __init__(self):
        super(RecallLoss, self).__init__()
        self.sub_loss = nn.CrossEntropyLoss()

    def forward(self, preds, trues):
        preds = torch.softmax(preds, dim=1)
        one_hot = torch.zeros(preds.size(), device=conf.device_name)
        trues_oh = one_hot.scatter_(1, trues.view(-1, 1), 1)
        tp = preds * trues_oh
        recall = tp.sum(dim=0) / (trues_oh.sum(dim=0) + 1e-8)
        return recall.mean() + self.sub_loss(preds, trues)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0.5, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.4):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sin = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sin * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda:0')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma = 1
        loss = (loss1 + gamma*loss2) / (1+gamma)
        return loss


