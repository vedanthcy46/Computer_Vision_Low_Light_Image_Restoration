"""NIQE-style no-reference quality metric (Mittal et al., 2013), self-contained.

NIQE fits a multivariate Gaussian (MVG) to natural-scene-statistics (NSS) features of
*pristine* images and scores a test image by the distance between its MVG and the pristine
one (lower = more natural). The official model parameters are downloaded by ``pyiqa`` (needs
internet); this module implements the same feature pipeline and lets you *fit* the pristine
model on well-exposed images of your own dataset (e.g. the normal-light LOL images), which makes
the metric fully offline and domain-matched.  For numbers comparable with the literature use
``scripts/evaluate.py --niqe pyiqa``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.special import gamma as _gamma

from .utils import luminance

_GAM = np.arange(0.2, 10.0, 0.001)
_RGAM = _gamma(2.0 / _GAM) ** 2 / (_gamma(1.0 / _GAM) * _gamma(3.0 / _GAM))
DEFAULT_PARAMS = Path(__file__).resolve().parents[1] / "weights" / "niqe_params.npz"


def _aggd(x: np.ndarray):
    left, right = x[x < 0], x[x >= 0]
    ls = np.sqrt(np.mean(left ** 2)) if left.size else 1e-6
    rs = np.sqrt(np.mean(right ** 2)) if right.size else 1e-6
    gh = ls / (rs + 1e-12)
    rhat = np.mean(np.abs(x)) ** 2 / (np.mean(x ** 2) + 1e-12)
    rn = rhat * (gh ** 3 + 1) * (gh + 1) / ((gh ** 2 + 1) ** 2 + 1e-12)
    a = _GAM[np.argmin((_RGAM - rn) ** 2)]
    bl = ls * np.sqrt(_gamma(1 / a) / _gamma(3 / a))
    br = rs * np.sqrt(_gamma(1 / a) / _gamma(3 / a))
    return a, bl, br


def _mscn(y: np.ndarray) -> np.ndarray:
    mu = cv2.GaussianBlur(y, (0, 0), 7 / 6, borderType=cv2.BORDER_REFLECT)
    var = cv2.GaussianBlur(y * y, (0, 0), 7 / 6, borderType=cv2.BORDER_REFLECT) - mu * mu
    return (y - mu) / (np.sqrt(np.abs(var)) + 1.0)


def _patch_features(m: np.ndarray) -> np.ndarray:
    a, bl, br = _aggd(m.ravel())
    f = [a, (bl + br) / 2]
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        shifted = np.roll(np.roll(m, dy, 0), dx, 1)
        a, bl, br = _aggd((m * shifted).ravel())
        mean = (br - bl) * _gamma(2 / a) / _gamma(1 / a)
        f += [a, mean, bl, br]
    return np.array(f, dtype=np.float64)


def nss_features(img: np.ndarray, patch: int = 96) -> np.ndarray:
    """(n_patches, 36) NSS features over two scales."""
    y = luminance(img).astype(np.float32) * 255.0
    feats = []
    for scale in range(2):
        if scale:
            y = cv2.resize(y, (y.shape[1] // 2, y.shape[0] // 2), interpolation=cv2.INTER_AREA)
        p = patch // (2 ** scale)
        m = _mscn(y)
        h, w = m.shape
        rows = [(i, j) for i in range(0, h - p + 1, p) for j in range(0, w - p + 1, p)]
        feats.append(np.stack([_patch_features(m[i:i + p, j:j + p]) for i, j in rows]) if rows else None)
    if feats[0] is None or feats[1] is None:
        raise ValueError("image too small for NIQE (needs >= 192 px on the short side)")
    n = min(len(feats[0]), len(feats[1]))
    return np.concatenate([feats[0][:n], feats[1][:n]], axis=1)


class NIQE:
    def __init__(self, mu: np.ndarray, cov: np.ndarray):
        self.mu, self.cov = mu, cov

    @classmethod
    def fit(cls, images) -> "NIQE":
        f = np.concatenate([nss_features(im) for im in images if min(im.shape[:2]) >= 192])
        return cls(f.mean(0), np.cov(f, rowvar=False))

    def save(self, path=DEFAULT_PARAMS):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=self.mu, cov=self.cov)

    @classmethod
    def load(cls, path=DEFAULT_PARAMS) -> "NIQE | None":
        path = Path(path)
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["mu"], d["cov"])

    def score(self, img: np.ndarray) -> float:
        # Upscale small images so that at least a few 96px patches exist.
        s = min(img.shape[:2])
        if s < 192:
            f = 192 / s
            img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_CUBIC)
        f = nss_features(np.clip(img, 0, 1))
        mu, cov = f.mean(0), (np.cov(f, rowvar=False) if len(f) > 1 else np.zeros_like(self.cov))
        pooled = (self.cov + cov) / 2.0
        # ridge term keeps the solve stable when only a few patches were available
        pooled = pooled + np.eye(len(pooled)) * 1e-3 * (np.trace(pooled) / len(pooled) + 1e-9)
        d = self.mu - mu
        return float(np.sqrt(max(d @ np.linalg.solve(pooled, d), 0.0)))
