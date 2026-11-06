"""Image I/O and colour helpers.

Convention used everywhere in this package: images are ``float32`` RGB arrays of
shape ``(H, W, 3)`` with values in ``[0, 1]``.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EPS = 1e-6
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class InvalidImageError(ValueError):
    """Raised when an input cannot be interpreted as an image."""


def to_float(img: np.ndarray) -> np.ndarray:
    """Convert any numeric image (gray / RGB / RGBA, uint8 / uint16 / float) to float32 RGB."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise InvalidImageError("empty image")
    a = img
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    elif a.ndim != 3 or a.shape[2] != 3:
        raise InvalidImageError(f"unsupported image shape {img.shape}")
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        if a.max() > 1.0 + 1e-3:  # float image on a 0-255 scale
            a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_image(src, max_side: int | None = None) -> np.ndarray:
    """Load an image from a path / bytes / file-like object into float32 RGB.

    EXIF orientation is honoured. ``max_side`` optionally down-scales large inputs.
    """
    try:
        if isinstance(src, (str, Path)):
            im = Image.open(src)
        elif isinstance(src, (bytes, bytearray)):
            im = Image.open(io.BytesIO(src))
        else:
            im = Image.open(src)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I"):
            arr = np.asarray(im).astype(np.float32)
            arr = arr / (65535.0 if arr.max() > 255 else 255.0)
            img = to_float(arr)
        else:
            img = to_float(np.asarray(im.convert("RGB")))
    except InvalidImageError:
        raise
    except Exception as exc:  # corrupt / unsupported file
        raise InvalidImageError("unsupported or corrupt image file") from exc
    if max_side:
        img = resize_max_side(img, max_side)
    return img


def save_image(path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img)).save(path)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    f = max_side / s
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.709 luma of a float RGB image -> (H, W)."""
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def rgb2lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(img, dtype=np.float32), cv2.COLOR_RGB2LAB)


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB), 0.0, 1.0)


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def side_by_side(*imgs: np.ndarray) -> np.ndarray:
    h = min(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        w = int(round(i.shape[1] * h / i.shape[0]))
        out.append(cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA))
    return np.concatenate(out, axis=1)
