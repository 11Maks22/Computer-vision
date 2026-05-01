from __future__ import annotations

import torch
from torch import nn


class SRCNN(nn.Module):
    """A shallow convolutional network for super-resolution.

    The model learns a mapping from interpolated low-resolution input to
    high-resolution target. It is intentionally small and interpretable,
    which makes it suitable for technical coursework and controlled experiments.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, f1: int = 64, f2: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, out_channels, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
