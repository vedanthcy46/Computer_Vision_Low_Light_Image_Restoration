"""Fit the NIQE-style pristine model on well-exposed images (e.g. the normal-light LOL images).

    python scripts/fit_niqe.py data/LOL/our485/high data/LOL-v2/Real_captured/Train/Normal
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight.data import bundled_images, list_images  # noqa: E402
from lowlight.niqe import DEFAULT_PARAMS, NIQE  # noqa: E402
from lowlight.utils import load_image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="*", help="folders of well-exposed ('pristine') images")
ap.add_argument("--max", type=int, default=300)
ap.add_argument("--out", default=str(DEFAULT_PARAMS))
a = ap.parse_args()
if a.dirs:
    paths = [p for d in a.dirs for p in list_images(d)][: a.max]
    imgs = [load_image(p) for p in paths]
else:
    print("no folders given - fitting on the small built-in image set (demo only)")
    imgs = bundled_images(384)
NIQE.fit(imgs).save(a.out)
print(f"fitted on {len(imgs)} images -> {a.out}")
