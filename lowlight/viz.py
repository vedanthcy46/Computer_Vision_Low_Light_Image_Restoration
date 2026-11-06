"""Visualisation helpers for intermediate pipeline results."""
from __future__ import annotations

import cv2
import numpy as np

from .utils import luminance, to_uint8

NICE_NAMES = {
    "illumination": "Illumination map L (guided max-RGB)",
    "reflectance": "Reflectance R = I / L",
    "gain": "Illumination gain map (heat)",
    "retinex": "Retinex-brightened (before denoising)",
    "denoised": "Gain-aware edge-preserving denoise",
    "shrunk": "Noise-aware detail shrinkage",
    "tone": "Tone stretch",
    "clahe": "CLAHE (lightness)",
    "color": "Colour correction",
    "curves": "Zero-DCE curve output",
    "dl_branch": "Deep branch (denoise -> DCE -> shrink -> colour)",
    "classical_branch": "Classical branch output",
    "w_dl": "Fusion weight of the deep branch (heat)",
    "fused": "Pyramid-fused result",
    "final": "Final output",
}


def to_display(name: str, arr: np.ndarray) -> np.ndarray:
    """float array -> displayable float RGB. 2-D maps are normalised and colour-mapped."""
    if arr.ndim == 2:
        a = arr.astype(np.float32)
        if name == "illumination":
            return np.repeat(np.clip(a, 0, 1)[..., None], 3, axis=2)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / max(hi - lo, 1e-6)
        heat = cv2.applyColorMap(to_uint8(a), cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(arr, 0, 1)


def histogram_figure(images: dict[str, np.ndarray]):
    """Luma histograms for several images (matplotlib figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for name, im in images.items():
        ax.hist(luminance(im).ravel(), bins=64, range=(0, 1), histtype="step", label=name, density=True)
    ax.set_xlabel("luma")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
