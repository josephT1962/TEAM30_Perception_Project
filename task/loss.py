import torch
from torch import nn
import time
import numpy as np
from utils.misc import make_input


class DummyLoss(torch.nn.Module):
    def __init__(self):
        super(DummyLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        l = self.loss(pred, gt)
        return l.mean()


