"""Batch inference CLI: enhance every image in a folder.

    python run_inference.py --input_dir ./samples/ --output_dir ./results/
    python run_inference.py --input_dir ./samples/ --output_dir ./results/ --method classical --max_side 1200

Methods: hybrid (default, needs weights/zero_dce.pt), classical, deep. Unreadable files are reported and skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lowlight.classical import ClassicalConfig, ClassicalEnhancer
from lowlight.data import list_images
from lowlight.dl.infer import DLEnhancer, load_model
from lowlight.hybrid import HybridConfig, HybridEnhancer
from lowlight.utils import InvalidImageError, load_image, save_image


def build(method: str):
    if method == "classical":
        return ClassicalEnhancer(ClassicalConfig())
    model, loaded = load_model()
    if not loaded:
        sys.exit("weights/zero_dce.pt not found: train it (see README) or use --method classical")
    dl = DLEnhancer(model=model)
    return dl if method == "deep" else HybridEnhancer(dl, HybridConfig())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--method", choices=["hybrid", "classical", "deep"], default="hybrid")
    ap.add_argument("--max_side", type=int, default=1200, help="down-scale larger inputs to this max side (px)")
    a = ap.parse_args()

    files = list_images(a.input_dir)
    if not files:
        sys.exit(f"no images found in {a.input_dir}")
    enhancer = build(a.method)
    out_dir = Path(a.output_dir)
    done, ms_total = 0, 0.0
    for p in files:
        try:
            img = load_image(p, max_side=a.max_side)
        except InvalidImageError as e:
            print(f"SKIP {p.name}: {e}")
            continue
        t = time.perf_counter()
        res = enhancer.enhance(img)
        ms = (time.perf_counter() - t) * 1000.0
        save_image(out_dir / f"{p.stem}_{a.method}.png", res.image)
        done, ms_total = done + 1, ms_total + ms
        print(f"OK   {p.name}: {img.shape[1]}x{img.shape[0]} px, {ms:.0f} ms, "
              f"darkness {res.stats.darkness:.2f}, noise {res.stats.noise_level:.2f}")
    if done:
        print(f"{done}/{len(files)} images enhanced, mean {ms_total / done:.0f} ms/image -> {out_dir}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
