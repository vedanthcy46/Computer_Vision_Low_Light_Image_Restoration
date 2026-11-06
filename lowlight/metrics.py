"""Evaluation metrics: full-reference, no-reference, noise / edge / exposure / colour diagnostics, latency."""
from __future__ import annotations

import time
from functools import lru_cache

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .niqe import NIQE
from .quality import estimate_noise_sigma
from .utils import EPS, luminance, rgb2lab

# Column order used by every results table (matches the order in the problem statement).
METRIC_ORDER = ["PSNR", "SSIM", "LPIPS", "NIQE", "Latency_ms", "NoiseSuppression", "EdgePreservation"]
EXTRA_ORDER = ["Overexposure", "ColourShift_dE", "DetailRatio", "MeanLuma"]


# ---------------------------------------------------------------- full reference
def psnr(ref: np.ndarray, img: np.ndarray) -> float:
    err = float(np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2))
    return 100.0 if err < 1e-10 else float(10 * np.log10(1.0 / err))


def ssim(ref: np.ndarray, img: np.ndarray) -> float:
    return float(structural_similarity(ref, img, channel_axis=2, data_range=1.0))


@lru_cache(maxsize=1)
def _lpips_model():
    try:
        import warnings

        import lpips
        import torch
        warnings.filterwarnings("ignore", category=UserWarning)
        m = lpips.LPIPS(net="alex", verbose=False).eval()
        return m, torch
    except Exception:  # weights not downloadable / package missing
        return None


def lpips_score(ref: np.ndarray, img: np.ndarray) -> float:
    """LPIPS (AlexNet). Returns NaN if the pretrained backbone cannot be loaded."""
    mm = _lpips_model()
    if mm is None:
        return float("nan")
    m, torch = mm
    t = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1  # noqa: E731
    with torch.no_grad():
        return float(m(t(ref), t(img)).item())


def lpips_available() -> bool:
    return _lpips_model() is not None


# -------------------------------------------------------------- no reference
@lru_cache(maxsize=1)
def _niqe_model():
    return NIQE.load()


@lru_cache(maxsize=1)
def _pyiqa_niqe():
    try:
        import pyiqa
        import torch
        return pyiqa.create_metric("niqe", device="cpu"), torch
    except Exception:  # package missing or model parameters not downloadable
        return None


def niqe_score(img: np.ndarray, backend: str = "auto") -> float:
    """Naturalness (lower is better).

    ``pyiqa``: official NIQE parameters; ``local``: self-fitted NIQE-style model
    (``weights/niqe_params.npz``, see ``scripts/fit_niqe.py``); ``auto``: local if fitted, else pyiqa.
    """
    if backend == "auto" and _niqe_model() is not None:
        backend = "local"            # offline + fast; pyiqa only if no local model was fitted
    if backend in ("auto", "pyiqa"):
        mm = _pyiqa_niqe()
        if mm is not None:
            m, torch = mm
            return float(m(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float()).item())
        if backend == "pyiqa":
            raise RuntimeError("pyiqa NIQE is unavailable (package or weights could not be loaded)")
    m = _niqe_model()
    return m.score(img) if m is not None else float("nan")


# ------------------------------------------------------------------ diagnostics
def noise_suppression(inp: np.ndarray, out: np.ndarray) -> float:
    """Ratio of noise std of the *naively brightened input* to that of the output (higher = cleaner).

    The input is linearly scaled to the output's mean luma so both are compared at equal brightness;
    noise is the robust MAD / Laplacian estimate (insensitive to edges).
    """
    g = luminance(out).mean() / (luminance(inp).mean() + EPS)
    naive = np.clip(inp * g, 0, 1)
    return float(estimate_noise_sigma(naive) / (estimate_noise_sigma(out) + 1e-4))


def edge_preservation(ref: np.ndarray, out: np.ndarray) -> float:
    """Edge-preservation index: correlation of high-pass (Laplacian of lightly smoothed luma)
    between reference and output (Sattar et al.). 1 = edges perfectly preserved."""
    def hp(x):
        y = cv2.GaussianBlur(luminance(x), (0, 0), 1.0)
        return cv2.Laplacian(y, cv2.CV_32F)
    a, b = hp(ref), hp(out)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + EPS))


def overexposure(out: np.ndarray, ref: np.ndarray | None = None, thr: float = 0.98) -> float:
    """Fraction of (nearly) saturated pixels in the output, minus the reference's own (>=0)."""
    f = float((out.max(axis=2) > thr).mean())
    if ref is not None:
        f = max(0.0, f - float((ref.max(axis=2) > thr).mean()))
    return f


def colour_shift(out: np.ndarray, ref: np.ndarray) -> float:
    """Mean chromatic difference (dE on the Lab a,b plane, lightness ignored) to the reference."""
    a, b = rgb2lab(out)[..., 1:], rgb2lab(ref)[..., 1:]
    return float(np.sqrt(((a - b) ** 2).sum(-1)).mean())


def detail_ratio(out: np.ndarray, ref: np.ndarray) -> float:
    """High-frequency energy of output relative to reference (1 = same amount of detail)."""
    def e(x):
        y = luminance(x)
        return float((y - cv2.GaussianBlur(y, (0, 0), 2.0)).std())
    return e(out) / (e(ref) + EPS)


# --------------------------------------------------------------------- latency
def measure_latency(fn, img: np.ndarray, repeat: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn(img)
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn(img)
        ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts))


# ---------------------------------------------------------------------- bundle
def evaluate_pair(inp: np.ndarray, out: np.ndarray, gt: np.ndarray | None = None,
                  latency_ms: float = float("nan"), niqe_backend: str = "auto") -> dict:
    """All metrics for one enhanced image. Full-reference metrics are NaN when ``gt`` is None."""
    ref = gt if gt is not None else None
    nan = float("nan")
    ref_edge = gt if gt is not None else np.clip(inp * (luminance(out).mean() / (luminance(inp).mean() + EPS)), 0, 1)
    return {
        "PSNR": psnr(gt, out) if gt is not None else nan,
        "SSIM": ssim(gt, out) if gt is not None else nan,
        "LPIPS": lpips_score(gt, out) if gt is not None else nan,
        "NIQE": niqe_score(out, niqe_backend),
        "Latency_ms": latency_ms,
        "NoiseSuppression": noise_suppression(inp, out),
        "EdgePreservation": edge_preservation(ref_edge, out),
        "Overexposure": overexposure(out, ref),
        "ColourShift_dE": colour_shift(out, gt) if gt is not None else nan,
        "DetailRatio": detail_ratio(out, ref_edge),
        "MeanLuma": float(luminance(out).mean()),
    }
