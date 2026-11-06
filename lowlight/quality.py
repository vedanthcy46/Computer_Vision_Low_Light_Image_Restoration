"""Image-quality analysis that drives every adaptive decision in the pipelines."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .utils import EPS, luminance, smoothstep, to_float

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Robust (MAD-based) Immerkaer noise estimate on the luminance, in [0,1] intensity units.

    The Laplacian-difference kernel has noise gain sqrt(36) = 6, and the median
    absolute value of a zero-mean Gaussian is 0.6745*sigma. The median makes the
    estimate insensitive to edges and texture.
    """
    y = luminance(img) if img.ndim == 3 else img
    if min(y.shape) < 4:
        return 0.0
    r = cv2.filter2D(y.astype(np.float32), -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.median(np.abs(r)) / 0.6745 / 6.0)


@dataclass
class ImageStats:
    mean_lum: float          # mean luma
    mean_max: float          # mean of max(R,G,B): proxy for the illumination map
    p1: float
    p5: float
    p50: float
    p95: float
    p99: float
    contrast: float          # luma standard deviation
    dark_frac: float         # fraction of pixels with luma < 0.1
    saturated_frac: float    # fraction of pixels with any channel > 0.98
    noise_sigma: float
    noise_level: float       # 0..1
    non_uniformity: float    # 0..1, how uneven the lighting is
    color_cast: float        # max |channel_mean / grey_mean - 1|
    darkness: float          # 0..1, 1 = extremely under-exposed

    def as_dict(self) -> dict:
        return {k: float(v) for k, v in asdict(self).items()}


def analyze(img: np.ndarray) -> ImageStats:
    """Compute exposure / noise / lighting / colour statistics of a float RGB image."""
    img = to_float(img)
    y = luminance(img)
    mx = img.max(axis=2)
    p1, p5, p50, p95, p99 = (float(v) for v in np.percentile(y, [1, 5, 50, 95, 99]))
    mean_lum = float(y.mean())

    sigma = estimate_noise_sigma(img)
    noise_level = float(np.clip((sigma - 0.003) / 0.025, 0.0, 1.0))

    # Lighting non-uniformity: dynamic range of the heavily blurred log-luminance.
    h, w = y.shape
    k = max(3, int(max(h, w) / 16) | 1)
    small = cv2.resize(y, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), max(1.0, k / 8.0))
    lg = np.log2(blur + 0.01)
    non_uniformity = float(np.clip((np.percentile(lg, 95) - np.percentile(lg, 5)) / 4.0, 0.0, 1.0))

    m = img.reshape(-1, 3).mean(axis=0)
    cast = float(np.max(np.abs(m / (m.mean() + EPS) - 1.0)))

    darkness = float(1.0 - smoothstep(np.float32(mean_lum), 0.03, 0.45))
    return ImageStats(
        mean_lum=mean_lum, mean_max=float(mx.mean()), p1=p1, p5=p5, p50=p50, p95=p95, p99=p99,
        contrast=float(y.std()), dark_frac=float((y < 0.1).mean()),
        saturated_frac=float((mx > 0.98).mean()), noise_sigma=sigma, noise_level=noise_level,
        non_uniformity=non_uniformity, color_cast=cast, darkness=darkness,
    )
