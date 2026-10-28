"""DSen2-CR-style baseline (Meraner et al., ISPRS 2020): a residual ResNet on [S1, cloudy S2].

This is a re-implementation for in-house comparison, not the authors' code. The paper uses
256 features and 16 blocks at full resolution, which is too heavy for a Colab T4 at our crop
sizes; the defaults below are smaller, so do not quote the paper's numbers for this model.
"""
import torch
import torch.nn as nn

from ..bands import N_S1, N_S2


class ResBlock(nn.Module):
    def __init__(self, ch, scale=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, padding=1))
        self.scale = scale

    def forward(self, x):
        return x + self.scale * self.body(x)


class DSen2CR(nn.Module):
    def __init__(self, features=128, blocks=8):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(N_S1 + N_S2, features, 3, padding=1), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResBlock(features) for _ in range(blocks)])
        self.out = nn.Conv2d(features, N_S2, 3, padding=1)

    def forward(self, cloudy, s1):
        return cloudy + self.out(self.body(self.inp(torch.cat([s1, cloudy], 1))))
