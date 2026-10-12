"""A compact 2D U-Net that reads a stack of surface slices as input channels ("2.5D").

Ink is a subtle texture change through the depth of the papyrus, so the slices stack along the
channel axis and the network learns which depths matter. Kept small on purpose (a few million
parameters) so it trains in minutes on one GPU; tile sizes must be a multiple of 16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class InkUNet(nn.Module):
    def __init__(self, in_channels: int = 16, base: int = 32):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        prev = in_channels
        for w in widths:
            self.enc.append(_block(prev, w))
            prev = w
        self.bottleneck = _block(widths[-1], widths[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(prev, w, 2, stride=2))
            self.dec.append(_block(w * 2, w))
            prev = w
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x).squeeze(1)  # logits, (B, H, W)
