"""Named method / ablation variants shared by the evaluation script, tests and the app."""
from __future__ import annotations

import cv2
import numpy as np

from .classical import ClassicalConfig, ClassicalEnhancer
from .dl.infer import DLEnhancer
from .hybrid import HybridConfig, HybridEnhancer
from .utils import luminance, rgb2lab, lab2rgb


class _Fn:
    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    def __call__(self, img):
        return self._fn(img)


def identity():
    return _Fn("input", lambda x: x)


def auto_gamma():
    """Naive baseline: global gamma that maps the mean luma to 0.45."""
    def f(x):
        m = float(np.clip(luminance(x).mean(), 0.01, 0.99))
        g = float(np.clip(np.log(0.45) / np.log(m), 0.2, 1.0))
        return np.clip(x ** g, 0, 1).astype(np.float32)
    return _Fn("gamma", f)


def hist_eq():
    """Naive baseline: global histogram equalisation of the Lab lightness channel."""
    def f(x):
        lab = rgb2lab(x)
        l8 = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
        lab[..., 0] = cv2.equalizeHist(l8).astype(np.float32) / 2.55
        return lab2rgb(lab)
    return _Fn("hist_eq", f)


def main_methods(dl: DLEnhancer) -> dict:
    """The three approaches required by the challenge (+ two naive baselines for reference)."""
    return {
        "input": identity(),
        "gamma": auto_gamma(),
        "hist_eq": hist_eq(),
        "classical": ClassicalEnhancer(ClassicalConfig()),
        "deep_learning": dl,
        "hybrid": HybridEnhancer(dl, HybridConfig()),
    }


def ablation_variants(dl: DLEnhancer) -> dict:
    """Each row removes / swaps exactly one component of the full system."""
    H = lambda **kw: HybridEnhancer(dl, HybridConfig(**kw))  # noqa: E731
    HC = lambda **kw: HybridEnhancer(dl, HybridConfig(classical=ClassicalConfig(**kw)))  # noqa: E731
    C = lambda **kw: ClassicalEnhancer(ClassicalConfig(**kw))  # noqa: E731
    return {
        # --- hybrid system ------------------------------------------------------------
        "hybrid (full)": H(),
        "hybrid - adaptive parameters (fixed)": H(adaptive=False),
        "hybrid - illumination prior": H(use_illum_prior=False),
        "hybrid - pyramid blending (alpha blend)": H(use_pyramid=False),
        "hybrid - DL pre-denoise": H(use_denoise=False),
        "hybrid - DL detail shrinkage": H(use_shrink=False),
        "hybrid - output guard (highlights/exposure)": H(use_guard=False),
        "hybrid - classical denoise": HC(use_denoise=False),
        "hybrid - classical shrinkage": HC(use_shrink=False),
        "hybrid - tone stretch": HC(use_tone=False),
        "hybrid - CLAHE": HC(use_clahe=False),
        "hybrid - colour correction": HC(use_color=False),
        "hybrid - deep branch (= classical only)": H(use_dl=False),
        "hybrid - classical branch (= DL + post only)": H(use_classical=False),
        # --- classical pipeline -------------------------------------------------------
        "classical (full)": C(),
        "classical - adaptive parameters (fixed)": C(adaptive=False),
        "classical - denoise": C(use_denoise=False),
        "classical - shrinkage": C(use_shrink=False),
        "classical - tone stretch": C(use_tone=False),
        "classical - CLAHE": C(use_clahe=False),
        "classical - colour correction": C(use_color=False),
        "classical - guard": C(use_guard=False),
        "classical: bilateral instead of guided denoise": C(denoise_mode="bilateral"),
        "classical: MSRCP instead of illumination decomposition": C(retinex="msr"),
        # --- deep learning ------------------------------------------------------------
        "deep learning (quality-aware)": dl,
        "deep learning - quality-awareness (raw Zero-DCE)": DLEnhancer(model=dl.model, quality_aware=False),
    }
