"""Datasets: paired / unpaired discovery for LOL, LOL-v2 and FiveK, synthetic low-light
generation, and PyTorch datasets for training."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .utils import IMG_EXTS, load_image, to_float

LOW_DIRS = {"low", "lowlight", "input", "dark", "lq"}
HIGH_DIRS = {"high", "normal", "gt", "reference", "target", "expertc", "hq"}


def list_images(d: str | Path) -> list[Path]:
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())


def _key(p: Path) -> str:
    return re.sub(r"(low|normal|high|gt|dark)[_\-]?", "", p.stem.lower())


def find_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Find (low, ground-truth) pairs below ``root``.

    Works for LOL (``low/`` + ``high/``), LOL-v2 (``Low/`` + ``Normal/``, files ``low00001.png`` /
    ``normal00001.png``) and any folder pair using the usual names. Pairs are matched by file stem
    after stripping the ``low``/``normal``/``high``/``gt`` prefix.
    """
    root = Path(root)
    pairs: list[tuple[Path, Path]] = []
    dirs = [root] + [d for d in root.rglob("*") if d.is_dir()]
    for d in dirs:
        subs = {s.name.lower(): s for s in d.iterdir() if s.is_dir()}
        low = next((subs[n] for n in subs if n in LOW_DIRS), None)
        high = next((subs[n] for n in subs if n in HIGH_DIRS), None)
        if low is None or high is None:
            continue
        gt = {_key(p): p for p in list_images(high)}
        for lp in list_images(low):
            if _key(lp) in gt:
                pairs.append((lp, gt[_key(lp)]))
    return pairs


# --------------------------------------------------------------------- synthetic data
def bundled_images(size: int = 256) -> list[np.ndarray]:
    """Natural images shipped with scikit-image (no download) - used for tests / smoke runs."""
    from skimage import data as skd
    names = ["astronaut", "coffee", "chelsea", "rocket", "retina", "motorcycle_left", "cat"]
    out = []
    for n in names:
        try:
            im = to_float(getattr(skd, n)())
        except Exception:
            continue
        h, w = im.shape[:2]
        s = size / min(h, w)
        out.append(cv2.resize(im, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA))
    return out


def degrade_low_light(img: np.ndarray, rng: np.random.Generator | None = None,
                      severity: float | None = None, non_uniform: bool = True) -> np.ndarray:
    """Physically-motivated low-light degradation of a well-exposed sRGB image.

    sRGB -> linear, spatially varying attenuation (optionally with a bright light-source blob),
    per-channel colour cast, signal-dependent shot noise + read noise, back to sRGB, 8-bit quantisation.
    ``severity`` in [0,1]: 0 = mildly dark, 1 = extremely dark.
    """
    rng = rng or np.random.default_rng()
    img = to_float(img)
    h, w = img.shape[:2]
    sev = float(rng.uniform(0.2, 1.0) if severity is None else severity)
    lin = img ** 2.2
    gain = 0.25 * (0.04 / 0.25) ** sev  # 0.25 .. 0.04
    field = np.ones((h, w), np.float32)
    if non_uniform:
        n = cv2.resize(rng.standard_normal((4, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        field = np.exp(0.6 * n)
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((yy - cy) / (0.25 * h)) ** 2 + ((xx - cx) / (0.25 * w)) ** 2))
        field = field / field.mean() * (0.5 + 3.0 * blob)
    dark = lin * (gain * field)[..., None]
    cast = rng.uniform(0.85, 1.15, 3).astype(np.float32)
    dark = dark * cast
    shot, read = rng.uniform(0.001, 0.004), rng.uniform(0.0003, 0.001)
    noisy = dark + np.sqrt(np.maximum(shot * dark + read ** 2, 0)) * rng.standard_normal(dark.shape)
    out = np.clip(noisy, 0, 1) ** (1 / 2.2)
    return (np.round(out * 255) / 255).astype(np.float32)


def make_synthetic_pairs(n: int = 12, size: int = 256, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = bundled_images(size)
    pairs = []
    for i in range(n):
        im = base[i % len(base)]
        h, w = im.shape[:2]
        y, x = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
        gt = im[y:y + size, x:x + size]
        if rng.random() < 0.5:
            gt = gt[:, ::-1].copy()
        pairs.append((degrade_low_light(gt, rng), gt))
    return pairs


# ------------------------------------------------------------------ torch datasets
class PatchDataset:
    """Random crops from images (``low_dir`` unpaired, or ``pairs`` for supervised fine-tuning)."""

    def __init__(self, low_paths: list[Path], gt_paths: list[Path] | None = None, patch: int = 256,
                 length_mult: int = 4, synthetic_from: list[np.ndarray] | None = None, seed: int = 0):
        self.low, self.gt, self.patch = low_paths, gt_paths, patch
        self.mult, self.syn = length_mult, synthetic_from
        self.rng = np.random.default_rng(seed)
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self):
        n = len(self.syn) if self.syn is not None else len(self.low)
        return n * self.mult

    def _load(self, p: Path):
        if p not in self._cache:
            self._cache[p] = load_image(p)
        return self._cache[p]

    def __getitem__(self, i):
        import torch
        n = len(self.syn) if self.syn is not None else len(self.low)
        i = i % n
        if self.syn is not None:
            gt = self.syn[i]
            low = None
        else:
            low = self._load(self.low[i])
            gt = self._load(self.gt[i]) if self.gt else None
        ref = gt if low is None else low
        h, w = ref.shape[:2]
        p = min(self.patch, h, w)
        y, x = int(self.rng.integers(0, h - p + 1)), int(self.rng.integers(0, w - p + 1))
        flip = self.rng.random() < 0.5
        crop = lambda a: (a[y:y + p, x:x + p][:, ::-1] if flip else a[y:y + p, x:x + p])  # noqa: E731
        if low is None:  # synthetic: degrade on the fly
            g = crop(gt)
            lo = degrade_low_light(np.ascontiguousarray(g), self.rng)
            gt_c = np.ascontiguousarray(g)
        else:
            lo = np.ascontiguousarray(crop(low))
            gt_c = np.ascontiguousarray(crop(gt)) if gt is not None else lo
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)).astype(np.float32))  # noqa: E731
        return t(lo), t(gt_c)
