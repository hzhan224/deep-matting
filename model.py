import torch
import torch.nn as nn

from modules import AutoEncoder

class MattingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = AutoEncoder()

    def forward(self, x):
        x = self.ae(x)
        return x