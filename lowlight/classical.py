"""Classical computer-vision low-light enhancement pipeline.

Stages (each can be switched off for the ablation study):

1. Image analysis + adaptive parameter selection
2. Illumination estimation and Retinex decomposition  I = R * L
3. Illumination adjustment L' = L**gamma  (capped per-pixel gain)
4. Gain-aware edge-preserving denoising (guided / bilateral, chroma-aware)
5. Noise-aware detail shrinkage (Wiener) - keeps edges, suppresses amplified noise,
   then black/white-point tone stretch
6. CLAHE on the lightness channel
7. Colour correction (partial grey-world + saturation compensation)
8. Soft highlight roll-off (over-exposure guard) and strength blending
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .adaptive import EnhanceParams, fixed_params, select_params
from .filters import denoise, soft_highlight_compress, wiener_shrink
from .quality import ImageStats, analyze
from .retinex import decompose, estimate_illumination, illumination_gain, msr_chroma_preserving
from .utils import lab2rgb, luminance, rgb2lab, to_float


@dataclass
class ClassicalConfig:
    adaptive: bool = True
    retinex: str = "decomposition"   # "decomposition" | "msr"
    use_denoise: bool = True
    denoise_mode: str = "guided"     # "guided" | "bilateral"
    use_shrink: bool = True
    use_tone: bool = True
    use_clahe: bool = True
    use_color: bool = True
    use_guard: bool = True


@dataclass
class EnhanceResult:
    image: np.ndarray
    stats: ImageStats
    params: EnhanceParams
    stages: dict[str, np.ndarray] = field(default_factory=dict)


def apply_clahe(img: np.ndarray, clip: float, grid: int, blend: float) -> np.ndarray:
    """CLAHE on the Lab lightness channel (16-bit precision), blended with the input lightness."""
    lab = rgb2lab(img)
    l16 = np.clip(lab[..., 0] / 100.0 * 65535.0, 0, 65535).astype(np.uint16)
    g = int(max(2, min(grid, min(img.shape[:2]) // 16)))
    cl = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(g, g)).apply(l16)
    lab[..., 0] = (1 - blend) * lab[..., 0] + blend * (cl.astype(np.float32) / 65535.0 * 100.0)
    return lab2rgb(lab)


def tone_stretch(img: np.ndarray, amount: float = 1.0, lo_pct: float = 0.5, hi_pct: float = 99.7) -> np.ndarray:
    """Common black/white-point stretch on luminance percentiles (colour ratios are preserved)."""
    y = luminance(img)
    lo, hi = (float(v) for v in np.percentile(y, [lo_pct, hi_pct]))
    if hi - lo < 0.05:
        return img
    lo *= 0.85 * amount
    hi = hi + (1.0 - hi) * (1.0 - amount)
    return np.clip((img - lo) / max(hi - lo, 1e-3), 0.0, 1.0).astype(np.float32)


def correct_colour(img: np.ndarray, strength: float, saturation: float) -> np.ndarray:
    """Partial grey-world white balance on non-saturated pixels + chroma gain in Lab."""
    out = img
    if strength > 1e-3:
        lum = luminance(img)
        mask = (lum > 0.05) & (lum < 0.9)
        if mask.sum() > 100:
            m = img[mask].mean(axis=0)
            gains = np.clip(m.mean() / (m + 1e-6), 0.75, 1.33)
            gains = 1.0 + strength * (gains - 1.0)
            out = np.clip(img * gains[None, None, :], 0.0, 1.0)
    if abs(saturation - 1.0) > 1e-3:
        lab = rgb2lab(out)
        lab[..., 1:] *= saturation
        out = lab2rgb(lab)
    return out.astype(np.float32)


class ClassicalEnhancer:
    name = "classical"

    def __init__(self, config: ClassicalConfig | None = None, **overrides):
        self.cfg = replace(config or ClassicalConfig(), **overrides)

    def enhance(self, img: np.ndarray, params: EnhanceParams | None = None,
                stats: ImageStats | None = None) -> EnhanceResult:
        cfg = self.cfg
        img = to_float(img)
        stats = stats or analyze(img)
        if params is None:
            params = select_params(stats) if cfg.adaptive else fixed_params()
            if not cfg.adaptive:
                params.noise_sigma = max(stats.noise_sigma, 1e-3)
        st: dict[str, np.ndarray] = {}

        # 2. illumination estimation & decomposition
        lum = estimate_illumination(img, radius=None)
        refl = np.clip(img / lum[..., None], 0, 1)
        st["illumination"], st["reflectance"] = lum, refl

        # 3. Retinex enhancement: brighten via the illumination curve L' = L**gamma
        if cfg.retinex == "msr":
            enh = msr_chroma_preserving(img)
            gain = np.clip(luminance(enh) / (luminance(img) + 1e-3), 1.0, params.gain_max * 2)
        else:
            gain = illumination_gain(lum, params.gamma, params.gain_max)
            enh = np.clip(img * gain[..., None], 0.0, 1.0)
        st["gain"] = gain
        st["retinex"] = enh

        # 4. gain-aware edge-preserving denoising: amplified noise has std sigma * gain per pixel
        noise_map = params.noise_sigma * gain
        if cfg.use_denoise:
            enh = denoise(enh, noise_map, params.denoise_k, params.guided_radius, cfg.denoise_mode)
        st["denoised"] = enh

        # 5. noise-aware detail shrinkage
        if cfg.use_shrink:
            enh = wiener_shrink(enh, params.shrink_c * noise_map)
        st["shrunk"] = enh

        # 5b. tone stretch: Retinex gain compresses contrast -> restore black / white points
        if cfg.use_tone:
            enh = tone_stretch(enh, params.stretch)

        # 6. CLAHE
        if cfg.use_clahe:
            enh = apply_clahe(enh, params.clahe_clip, params.clahe_grid, params.clahe_blend)
        st["clahe"] = enh

        # 7. colour correction
        if cfg.use_color:
            enh = correct_colour(enh, params.color_strength, params.saturation)
        st["color"] = enh

        # 8. over-exposure guard + adaptive strength
        if cfg.use_guard:
            enh = soft_highlight_compress(enh, params.highlight_knee)
            s = params.strength
            enh = (1 - s) * img + s * enh
        out = np.clip(enh, 0.0, 1.0).astype(np.float32)
        st["final"] = out
        return EnhanceResult(out, stats, params, st)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.enhance(img).image
