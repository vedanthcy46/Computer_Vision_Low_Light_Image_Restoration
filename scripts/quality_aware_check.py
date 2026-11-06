"""Compare the quality-aware deep branch against raw Zero-DCE on image sets (no ground truth needed).

    python scripts/quality_aware_check.py            # uses data/LOL/eval15/low and data/unseen/<set> if present

Reports mean input/output luma and over-exposed pixel fraction for both variants (see results/quality_aware_check.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight.data import list_images  # noqa: E402
from lowlight.dl.infer import DLEnhancer, load_model  # noqa: E402
from lowlight.metrics import overexposure  # noqa: E402
from lowlight.utils import load_image, resize_max_side  # noqa: E402

SETS = {"LOL eval15": "data/LOL/eval15/low", **{n: f"data/unseen/{n}" for n in ("DICM", "NPE", "VV", "Fusion", "MEF")}}


def main(n_images: int = 20, side: int = 400):
    model, _ = load_model()
    qa, raw = DLEnhancer(model=model, quality_aware=True), DLEnhancer(model=model, quality_aware=False)
    print("set | n | input luma | QA luma | raw luma | QA over-exp | raw over-exp | mean|QA-raw|")
    for name, rel in SETS.items():
        d = ROOT / rel
        if not d.exists():
            continue
        rows = []
        for p in list_images(d)[:n_images]:
            x = resize_max_side(load_image(p), side)
            a, b = qa(x), raw(x)
            rows.append((x.mean(), a.mean(), b.mean(), overexposure(a), overexposure(b), np.abs(a - b).mean()))
        print(f"{name} | {len(rows)} | " + " | ".join(f"{v:.3f}" for v in np.mean(rows, 0)))


if __name__ == "__main__":
    main()
