import math
import torch
import torch.nn.functional as F
from torch import nn


class Head(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super(Head, self).__init__()
        middle_size = max(output_size, input_size // 4)
        self.head = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.BatchNorm1d(middle_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size, output_size)
        )

    def forward(self, x):
        return self.head(x)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine

