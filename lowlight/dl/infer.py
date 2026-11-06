"""Deep-learning enhancer: Zero-DCE with an image-quality-aware strategy."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..adaptive import EnhanceParams, select_params
from ..classical import EnhanceResult
from ..filters import denoise, soft_highlight_compress
from ..quality import ImageStats, analyze
from ..utils import to_float
from .zero_dce import DCENet, apply_curves

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "zero_dce.pt"


def load_model(path: str | Path | None = None, device: str = "cpu") -> tuple[DCENet, bool]:
    """Load a DCENet checkpoint. Returns (model, weights_loaded)."""
    model = DCENet()
    path = Path(path or os.environ.get("LOWLIGHT_WEIGHTS") or DEFAULT_WEIGHTS)
    loaded = False
    if path.exists():
        ck = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        loaded = True
    return model.to(device).eval(), loaded


class DLEnhancer:
    """Zero-DCE inference.

    With ``quality_aware=True`` (the default) the strength of enhancement follows the
    measured image quality:

    * well-exposed images are left (almost) unchanged (no over-exposure),
    * the number of curve iterations scales with darkness,
    * a light denoise is applied first for noisy inputs (DCE amplifies noise),
    * an exposure-feedback step pulls back results that came out too bright.
    """
    name = "deep_learning"

    def __init__(self, model: DCENet | None = None, weights: str | Path | None = None,
                 quality_aware: bool = True, device: str = "cpu", max_side: int | None = None):
        if model is None:
            model, self.weights_loaded = load_model(weights, device)
        else:
            self.weights_loaded = True
        self.model, self.device = model.eval(), device
        self.quality_aware, self.max_side = quality_aware, max_side

    @torch.no_grad()
    def run_curves(self, img: np.ndarray, n_iter: int | None = None) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        maps = self.model.curve_maps(x)
        out = apply_curves(x, maps, n_iter)
        return out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float32)

    def enhance(self, img: np.ndarray, params: EnhanceParams | None = None,
                stats: ImageStats | None = None) -> EnhanceResult:
        img = to_float(img)
        stats = stats or analyze(img)
        params = params or select_params(stats)
        st: dict[str, np.ndarray] = {}
        x = img
        if self.quality_aware:
            if stats.noise_level > 0.15:
                x = denoise(img, params.noise_sigma, params.denoise_k * 0.7, params.guided_radius)
            n_iter = int(np.clip(round(2 + 6 * stats.darkness), 2, 8))
        else:
            n_iter = None
        st["denoised"] = x
        enh = self.run_curves(x, n_iter)
        st["curves"] = enh
        if self.quality_aware:
            enh = soft_highlight_compress(enh, params.highlight_knee)
            # exposure feedback: never end up much brighter than the target exposure
            mean_out = float(enh.mean())
            if mean_out > 0.62:
                w = float(np.clip((mean_out - 0.62) / 0.3, 0.0, 0.8))
                enh = (1 - w) * enh + w * img
            enh = (1 - params.strength) * img + params.strength * enh
        out = np.clip(enh, 0, 1).astype(np.float32)
        st["final"] = out
        return EnhanceResult(out, stats, params, st)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.enhance(img).image
