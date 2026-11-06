"""High-level facade used by the app and the evaluation scripts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .classical import ClassicalConfig, ClassicalEnhancer, EnhanceResult
from .dl.infer import DLEnhancer
from .hybrid import HybridConfig, HybridEnhancer
from .utils import to_float

METHODS = ("hybrid", "classical", "deep_learning")


@lru_cache(maxsize=2)
def get_dl(weights: str | None = None) -> DLEnhancer:
    return DLEnhancer(weights=weights)


def get_enhancer(method: str, weights: str | Path | None = None):
    if method == "classical":
        return ClassicalEnhancer(ClassicalConfig())
    dl = get_dl(str(weights) if weights else None)
    if method in ("deep_learning", "dl"):
        return dl
    if method == "hybrid":
        return HybridEnhancer(dl, HybridConfig())
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def enhance(img: np.ndarray, method: str = "hybrid", weights: str | Path | None = None) -> EnhanceResult:
    return get_enhancer(method, weights).enhance(to_float(img))
