"""Zero-reference losses of Zero-DCE plus a supervised loss for paired fine-tuning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """Keeps local contrast (neighbouring-region differences) of input and enhanced image equal."""

    def __init__(self):
        super().__init__()
        k = lambda a: torch.tensor(a, dtype=torch.float32).view(1, 1, 3, 3)  # noqa: E731
        self.register_buffer("l", k([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]))
        self.register_buffer("r", k([[0, 0, 0], [0, 1, -1], [0, 0, 0]]))
        self.register_buffer("u", k([[0, -1, 0], [0, 1, 0], [0, 0, 0]]))
        self.register_buffer("d", k([[0, 0, 0], [0, 1, 0], [0, -1, 0]]))
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enh):
        o = self.pool(org.mean(1, keepdim=True))
        e = self.pool(enh.mean(1, keepdim=True))
        loss = 0.0
        for k in (self.l, self.r, self.u, self.d):
            loss = loss + ((F.conv2d(o, k, padding=1) - F.conv2d(e, k, padding=1)) ** 2).mean()
        return loss


class ExposureLoss(nn.Module):
    def __init__(self, patch: int = 16, mean_val: float = 0.6):
        super().__init__()
        self.pool, self.mean_val = nn.AvgPool2d(patch), mean_val

    def forward(self, x):
        return ((self.pool(x.mean(1, keepdim=True)) - self.mean_val) ** 2).mean()


class ColorConstancyLoss(nn.Module):
    def forward(self, x):
        m = x.mean((2, 3))
        r, g, b = m[:, 0], m[:, 1], m[:, 2]
        return (((r - g) ** 2 + (r - b) ** 2 + (b - g) ** 2) ** 0.5).mean()


class IlluminationSmoothnessLoss(nn.Module):
    def forward(self, a):
        dh = (a[:, :, 1:, :] - a[:, :, :-1, :]) ** 2
        dw = (a[:, :, :, 1:] - a[:, :, :, :-1]) ** 2
        return dh.mean() + dw.mean()


class ZeroDCELoss(nn.Module):
    def __init__(self, w_spa=1.0, w_exp=10.0, w_col=5.0, w_tv=200.0, exp_level=0.6):
        super().__init__()
        self.spa, self.exp = SpatialConsistencyLoss(), ExposureLoss(16, exp_level)
        self.col, self.tv = ColorConstancyLoss(), IlluminationSmoothnessLoss()
        self.w = (w_spa, w_exp, w_col, w_tv)

    def forward(self, org, enh, maps):
        parts = {
            "spa": self.w[0] * self.spa(org, enh), "exp": self.w[1] * self.exp(enh),
            "col": self.w[2] * self.col(enh), "tv": self.w[3] * self.tv(maps),
        }
        return sum(parts.values()), parts


def _gauss_win(size=11, sigma=1.5, device="cpu"):
    ax = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim_torch(x, y, size=11):
    c = x.shape[1]
    w = _gauss_win(size, device=x.device).repeat(c, 1, 1, 1)
    f = lambda t: F.conv2d(t, w, padding=size // 2, groups=c)  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


class SupervisedLoss(nn.Module):
    """L1 + (1-SSIM) fidelity to the ground truth, plus the curve smoothness prior."""

    def __init__(self, w_l1=1.0, w_ssim=0.5, w_tv=20.0):
        super().__init__()
        self.w = (w_l1, w_ssim, w_tv)
        self.tv = IlluminationSmoothnessLoss()

    def forward(self, org, enh, maps, gt):
        parts = {
            "l1": self.w[0] * F.l1_loss(enh, gt),
            "ssim": self.w[1] * (1 - ssim_torch(enh, gt)),
            "tv": self.w[2] * self.tv(maps),
        }
        return sum(parts.values()), parts
