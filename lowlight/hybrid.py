"""Hybrid intelligence: classical illumination estimation + deep curve estimation.

Pipeline
--------
1. Analyse the image (exposure, noise, lighting non-uniformity, colour cast) and derive
   adaptive parameters.
2. Classical branch  : Retinex illumination decomposition, gain-capped brightening,
                       guided denoising, noise-aware detail shrinkage, CLAHE, colour correction.
3. Deep branch       : Zero-DCE curves on the (lightly) denoised input, then the same
                       noise-aware detail shrinkage and colour correction.
4. Adaptive fusion   : per-pixel weights from well-exposedness, (denoised) contrast and
                       saturation of each candidate, multiplied by an *illumination prior*
                       derived from the classical illumination map (very dark regions lean
                       on the explicitly noise-controlled classical branch), blended in a
                       Laplacian pyramid so no halos / seams appear.
5. Output guard      : soft highlight roll-off, exposure feedback, adaptive strength.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .adaptive import EnhanceParams, fixed_params, select_params
from .classical import ClassicalConfig, ClassicalEnhancer, EnhanceResult, correct_colour
from .dl.infer import DLEnhancer
from .filters import denoise, guided_filter, pyramid_fuse, soft_highlight_compress, wiener_shrink
from .quality import ImageStats, analyze
from .retinex import estimate_illumination
from .utils import luminance, smoothstep, to_float


@dataclass
class HybridConfig:
    adaptive: bool = True
    use_classical: bool = True
    use_dl: bool = True
    use_denoise: bool = True
    use_shrink: bool = True          # noise-aware detail shrinkage on the DL branch
    use_illum_prior: bool = True     # illumination-aware prior in the fusion weights
    use_pyramid: bool = True         # Laplacian-pyramid blending (False -> plain alpha blend)
    use_guard: bool = True
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)


def _well_exposed(y: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    return np.exp(-((y - 0.5) ** 2) / (2 * sigma ** 2))


def _contrast(y: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 1.2), cv2.CV_32F))


def _saturation(img: np.ndarray) -> np.ndarray:
    return img.std(axis=2)


class HybridEnhancer:
    name = "hybrid"

    def __init__(self, dl: DLEnhancer, config: HybridConfig | None = None):
        self.dl = dl
        self.cfg = config or HybridConfig()
        self.classical = ClassicalEnhancer(replace(self.cfg.classical, adaptive=self.cfg.adaptive))

    # ------------------------------------------------------------------ branches
    def _dl_branch(self, img, lum, params, stats, st):
        x = img
        if self.cfg.use_denoise:
            x = denoise(img, params.noise_sigma, params.denoise_k, params.guided_radius)
        n_iter = int(np.clip(round(2 + 6 * stats.darkness), 2, 8))
        d = self.dl.run_curves(x, n_iter)
        if self.cfg.use_shrink:
            gain = np.clip(luminance(d) / (luminance(x) + 1e-3), 1.0, params.gain_max * 2)
            d = wiener_shrink(d, params.shrink_c * params.noise_sigma * gain)
        d = correct_colour(d, params.color_strength, params.saturation)
        st["dl_branch"] = d
        return d

    def _weights(self, cands, lum, params):
        ws = []
        for i, c in enumerate(cands):
            y = luminance(c)
            w = (_contrast(y) + 1e-3) ** 0.6 * (_saturation(c) + 1e-3) ** 0.4 * (_well_exposed(y) + 1e-3) ** 1.0
            ws.append(w.astype(np.float32))
        if self.cfg.use_illum_prior and len(cands) == 2:
            mix = 0.4 + 0.6 * smoothstep(lum, 0.03, 0.2)       # dark -> less DL, bright -> more DL
            p_dl = np.clip(params.dl_bias * mix, 0.05, 0.95)
            ws[0] = ws[0] * (1 - p_dl)                          # classical
            ws[1] = ws[1] * p_dl                                # deep
        # smooth weights: avoids noisy / blocky selection masks
        return [guided_filter(luminance(cands[i]), w, 8, 1e-2) for i, w in enumerate(ws)]

    # ------------------------------------------------------------------- public
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
        lum = estimate_illumination(img)
        st["illumination"] = lum

        cands = []
        if cfg.use_classical:
            cp = replace(params, strength=1.0)
            cres = self.classical.enhance(img, cp, stats)
            st["classical_branch"] = cres.image
            st["gain"] = cres.stages["gain"]
            st["reflectance"] = cres.stages["reflectance"]
            st.update({f"classical/{k}": v for k, v in cres.stages.items() if k not in ("illumination", "reflectance", "gain", "final")})
            cands.append(cres.image)
        if cfg.use_dl:
            cands.append(self._dl_branch(img, lum, params, stats, st))

        if len(cands) == 1:
            fused = cands[0]
        else:
            ws = self._weights(cands, lum, params)
            tot = np.sum(ws, axis=0) + 1e-8
            st["w_dl"] = np.clip(ws[1] / tot, 0, 1)
            if cfg.use_pyramid:
                fused = pyramid_fuse(cands, ws)
            else:
                fused = np.clip(sum(c * (w / tot)[..., None] for c, w in zip(cands, ws)), 0, 1)
        st["fused"] = fused

        out = fused
        if cfg.use_guard:
            out = soft_highlight_compress(out, params.highlight_knee)
            mean_out = float(luminance(out).mean())
            if mean_out > 0.62:  # exposure feedback
                w = float(np.clip((mean_out - 0.62) / 0.3, 0.0, 0.8))
                out = (1 - w) * out + w * img
            out = (1 - params.strength) * img + params.strength * out
        out = np.clip(out, 0, 1).astype(np.float32)
        st["final"] = out
        return EnhanceResult(out, stats, params, st)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.enhance(img).image
