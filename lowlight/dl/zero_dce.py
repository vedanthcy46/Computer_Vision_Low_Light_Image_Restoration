"""Zero-DCE (Zero-Reference Deep Curve Estimation) in PyTorch.

Guo et al., CVPR 2020. A tiny CNN (~79k parameters) predicts per-pixel parameters of
8 iterations of a quadratic light-enhancement curve  LE(x) = x + a * x * (1 - x).
It can be trained without reference images (zero-reference losses) and fine-tuned
with paired data (supervised L1 + SSIM-style loss) when ground truth exists.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ITER = 8


class DCENet(nn.Module):
    def __init__(self, width: int = 32, n_iter: int = N_ITER):
        super().__init__()
        self.n_iter = n_iter
        c = width
        conv = lambda i, o: nn.Conv2d(i, o, 3, 1, 1, bias=True)  # noqa: E731
        self.c1, self.c2, self.c3, self.c4 = conv(3, c), conv(c, c), conv(c, c), conv(c, c)
        self.c5, self.c6 = conv(2 * c, c), conv(2 * c, c)
        self.c7 = conv(2 * c, 3 * n_iter)

    def curve_maps(self, x: torch.Tensor) -> torch.Tensor:
        r = F.relu
        x1 = r(self.c1(x))
        x2 = r(self.c2(x1))
        x3 = r(self.c3(x2))
        x4 = r(self.c4(x3))
        x5 = r(self.c5(torch.cat([x3, x4], 1)))
        x6 = r(self.c6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.c7(torch.cat([x1, x6], 1)))

    def forward(self, x: torch.Tensor, n_iter: int | None = None):
        """Returns (enhanced, curve_maps). ``n_iter`` < trained value gives a milder result."""
        maps = self.curve_maps(x)
        return apply_curves(x, maps, n_iter), maps


def apply_curves(x: torch.Tensor, maps: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
    total = maps.shape[1] // 3
    n = total if n_iter is None else int(min(n_iter, total))
    out = x
    for i in range(n):
        a = maps[:, 3 * i:3 * i + 3]
        out = out + a * (out * out - out)
    return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
