"""Adaptive parameter selection: image statistics -> enhancement parameters.

Instead of fixed constants, every knob of the pipeline is a smooth function of the
measured image quality (darkness, noise, lighting non-uniformity, colour cast).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .quality import ImageStats


@dataclass
class EnhanceParams:
    gamma: float = 0.5          # illumination curve exponent  (L' = L**gamma)
    gain_max: float = 10.0       # cap on the per-pixel illumination gain
    guided_radius: int = 4       # illumination-refinement / denoising radius
    denoise_k: float = 2.0       # denoise edge threshold in units of noise sigma
    shrink_c: float = 0.3        # Wiener-shrinkage noise multiplier
    clahe_clip: float = 2.0
    clahe_blend: float = 0.5
    clahe_grid: int = 8
    color_strength: float = 0.3  # partial grey-world correction strength
    saturation: float = 1.05
    stretch: float = 0.4         # tone-stretch amount
    highlight_knee: float = 0.88
    strength: float = 1.0        # overall enhancement strength (0 = leave image alone)
    dl_bias: float = 0.6         # hybrid: prior weight of the deep-learning branch
    noise_sigma: float = 0.01    # estimated sensor noise (used by denoising / shrinkage)

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in asdict(self).items()}


def fixed_params() -> EnhanceParams:
    """Non-adaptive defaults - used by the 'fixed parameters' ablation."""
    return EnhanceParams()


def select_params(stats: ImageStats, target: float = 0.42) -> EnhanceParams:
    """Derive enhancement parameters from measured image statistics."""
    m = float(np.clip(stats.mean_max, 0.02, 0.99))
    gamma = float(np.clip(np.log(target) / np.log(m), 0.4, 1.0))
    nl, dk = stats.noise_level, stats.darkness

    # Cap the illumination gain: noisy images must not be amplified as much.
    gain_max = 5.0 + 11.0 * (1.0 - nl)
    guided_radius = int(round(2 + 3 * nl))
    denoise_k = 1.2 + 2.0 * nl
    # shrinkage acts on the *residual* noise left after denoising, hence the small scale
    shrink_c = 0.2 + 0.3 * nl

    # CLAHE: stronger for dark, flat images; gentler when noisy (it amplifies noise).
    clahe_clip = float(np.clip(1.2 + 2.8 * dk * (1.0 - 0.6 * nl), 1.0, 4.0))
    clahe_blend = float(np.clip(0.25 + 0.55 * dk - 0.2 * nl, 0.1, 0.8))

    color_strength = float(0.6 * np.clip((stats.color_cast - 0.2) / 0.6, 0.0, 1.0))
    saturation = 1.0 + 0.15 * dk
    stretch = float(np.clip(0.15 + 0.35 * dk, 0.0, 0.5))

    # Overall strength: well-exposed images are left (almost) untouched -> no over-exposure.
    strength = float(np.clip(max(1.25 * dk, stats.dark_frac), 0.0, 1.0))
    dl_bias = float(np.clip(0.8 - 0.5 * nl, 0.25, 0.85))

    return EnhanceParams(
        gamma=gamma, gain_max=gain_max, guided_radius=guided_radius, denoise_k=denoise_k,
        shrink_c=shrink_c, clahe_clip=clahe_clip, clahe_blend=clahe_blend,
        color_strength=color_strength, saturation=saturation, stretch=stretch, strength=strength,
        dl_bias=dl_bias, noise_sigma=max(stats.noise_sigma, 1e-3),
    )
