"""Illumination estimation / decomposition and Retinex enhancement."""
from __future__ import annotations

import cv2
import numpy as np

from .filters import guided_filter
from .utils import EPS


def estimate_illumination(img: np.ndarray, radius: int | None = None, eps: float = 1e-3) -> np.ndarray:
    """Illumination map L (H,W) in (0,1].

    Initial estimate: per-pixel max over colour channels (the brightest channel bounds
    the incident light). It is refined with a guided filter (guide = the image's own
    max-channel), which smooths texture but keeps illumination edges (no halos), and is
    finally lower-bounded by the initial estimate so reflectance R = I/L stays <= 1.
    """
    h, w = img.shape[:2]
    l0 = img.max(axis=2).astype(np.float32)
    r = radius if radius is not None else int(np.clip(max(h, w) / 60, 4, 30))
    # Estimate on a down-scaled copy for speed on large images, then upsample.
    scale = 1.0
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        small = cv2.resize(l0, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        r = max(3, int(r * scale))
        ls = guided_filter(small, small, r, eps)
        l1 = cv2.resize(ls, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        l1 = guided_filter(l0, l0, r, eps)
    return np.clip(np.maximum(l1, l0), 1e-3, 1.0).astype(np.float32)


def decompose(img: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Retinex decomposition I = R * L. Returns (L, R)."""
    lum = estimate_illumination(img, **kw)
    refl = np.clip(img / lum[..., None], 0.0, 1.0)
    return lum, refl


def illumination_gain(lum: np.ndarray, gamma: float, gain_max: float) -> np.ndarray:
    """Per-pixel gain of the illumination curve L' = L**gamma, i.e. G = L'/L = L**(gamma-1)."""
    return np.clip(lum ** (gamma - 1.0), 1.0, gain_max).astype(np.float32)


def msr_chroma_preserving(img: np.ndarray, sigmas=(15, 80, 250), strength: float = 1.0) -> np.ndarray:
    """Multi-scale Retinex applied to intensity with chromaticity preserved (MSRCP).

    Log-ratio of the intensity to Gaussian surrounds at several scales, percentile-stretched,
    then channels are rescaled by the intensity gain so hue/saturation are retained.
    """
    inten = img.mean(axis=2) + 1.0 / 255.0
    msr = np.zeros_like(inten)
    for s in sigmas:
        msr += np.log(inten) - np.log(cv2.GaussianBlur(inten, (0, 0), s) + 1.0 / 255.0)
    msr /= len(sigmas)
    lo, hi = np.percentile(msr, [1, 99])
    out_i = np.clip((msr - lo) / max(hi - lo, EPS), 0.0, 1.0)
    gain = np.clip(out_i / (inten), 1.0, 12.0 * strength + 1.0 - strength)
    return np.clip(img * gain[..., None], 0.0, 1.0).astype(np.float32)
