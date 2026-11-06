"""Edge-preserving filters and multi-scale fusion building blocks (pure NumPy / OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

from .utils import EPS, luminance, rgb2lab, lab2rgb


def box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps) -> np.ndarray:
    """Gray-guide guided filter (He et al.). ``src`` may be (H,W) or (H,W,C).

    ``eps`` may be a scalar or an (H,W) map, which makes the smoothing threshold spatially
    adaptive (used for gain-aware denoising where amplified noise varies per pixel).
    """
    guide = guide.astype(np.float32)
    mean_i = box(guide, r)
    var_i = box(guide * guide, r) - mean_i * mean_i

    def _one(p: np.ndarray) -> np.ndarray:
        mean_p = box(p, r)
        cov_ip = box(guide * p, r) - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        return box(a, r) * guide + box(b, r)

    src = src.astype(np.float32)
    if src.ndim == 2:
        return _one(src)
    return np.stack([_one(src[..., c]) for c in range(src.shape[2])], axis=2)


def bilateral(img: np.ndarray, sigma_color: float, sigma_space: float = 5.0, d: int = 7) -> np.ndarray:
    return cv2.bilateralFilter(np.ascontiguousarray(img, dtype=np.float32), d, sigma_color, sigma_space)


def denoise(img: np.ndarray, sigma, k: float = 2.0, radius: int = 6, mode: str = "guided") -> np.ndarray:
    """Edge-preserving denoising with chroma smoothed more strongly than luma.

    ``sigma`` is the noise std (intensity units) of ``img`` - a scalar, or an (H,W) map when the
    image was brightened by a spatially varying gain. ``k`` scales the edge threshold: structures
    with std below ``k*sigma`` are treated as noise.
    """
    sigma = np.maximum(np.asarray(sigma, dtype=np.float32), 1e-4)
    if mode == "bilateral":
        sc = float(np.mean(sigma))
        out = bilateral(img, sigma_color=max(k * sc * 3.0, 0.02), sigma_space=float(radius), d=2 * min(radius, 5) + 1)
    else:
        guide = luminance(img)
        out = guided_filter(guide, img, radius, (k * sigma) ** 2)
    # Chroma denoising: chroma noise is the most objectionable artefact after amplification.
    lab = rgb2lab(np.clip(out, 0, 1))
    guide = lab[..., 0] / 100.0
    lab[..., 1:] = guided_filter(guide, lab[..., 1:], radius * 2, (k * sigma * 2.0) ** 2)
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def wiener_shrink(img: np.ndarray, noise_map: np.ndarray, radius: int = 8, win: int = 2) -> np.ndarray:
    """Noise-aware detail shrinkage.

    The image is split into base + detail. Where local detail energy barely exceeds the
    (spatially varying, post-amplification) noise variance ``noise_map**2`` the detail is
    attenuated; strong edges / textures (energy >> noise) pass through untouched. This
    keeps fine structure while preventing noise amplification in heavily brightened areas.
    """
    base = guided_filter(luminance(img), img, radius, 1e-3)
    detail = img - base
    energy = box((detail ** 2).mean(axis=2), win)
    var_n = (noise_map.astype(np.float32) ** 2)
    gain = np.clip((energy - var_n) / (energy + EPS), 0.0, 1.0)
    return np.clip(base + detail * gain[..., None], 0.0, 1.0)


def soft_highlight_compress(img: np.ndarray, knee: float = 0.88) -> np.ndarray:
    """Roll off values above ``knee`` smoothly instead of hard clipping (avoids blown highlights)."""
    knee = float(np.clip(knee, 0.5, 0.99))
    x = np.clip(img, 0.0, None)
    over = np.maximum(x - knee, 0.0)
    comp = knee + (1.0 - knee) * np.tanh(over / (1.0 - knee))
    return np.clip(np.where(x > knee, comp, x), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------- pyramid fusion
def _gauss_pyr(x: np.ndarray, levels: int) -> list[np.ndarray]:
    p = [x]
    for _ in range(levels - 1):
        p.append(cv2.pyrDown(p[-1]))
    return p


def _lap_pyr(x: np.ndarray, levels: int) -> list[np.ndarray]:
    g = _gauss_pyr(x, levels)
    lp = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lp.append(g[i] - up)
    lp.append(g[-1])
    return lp


def pyramid_fuse(images: list[np.ndarray], weights: list[np.ndarray], levels: int | None = None) -> np.ndarray:
    """Multi-resolution (Burt-Adelson / Mertens) blending of ``images`` with per-pixel ``weights``.

    Blending the Laplacian pyramids with Gaussian-pyramid weights avoids the halos and
    visible seams that plain per-pixel alpha blending produces at weight transitions.
    """
    h, w = images[0].shape[:2]
    if levels is None:
        levels = int(max(1, min(6, np.floor(np.log2(min(h, w) / 8)))))
    wsum = np.sum(weights, axis=0) + 1e-8
    wn = [(wt / wsum).astype(np.float32) for wt in weights]
    fused = None
    for im, wt in zip(images, wn):
        lp = _lap_pyr(im.astype(np.float32), levels)
        gp = _gauss_pyr(wt, levels)
        contrib = [l * g[..., None] for l, g in zip(lp, gp)]
        fused = contrib if fused is None else [f + c for f, c in zip(fused, contrib)]
    out = fused[-1]
    for i in range(levels - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(fused[i].shape[1], fused[i].shape[0])) + fused[i]
    return np.clip(out, 0.0, 1.0)
