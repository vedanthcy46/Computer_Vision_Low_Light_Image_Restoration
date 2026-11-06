"""Generate the figures used in report/ (side-by-side results and intermediate pipeline states).

    python scripts/make_report_figures.py     # needs weights/zero_dce.pt and the datasets under data/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight import metrics as M  # noqa: E402
from lowlight.classical import ClassicalConfig, ClassicalEnhancer  # noqa: E402
from lowlight.data import find_pairs, list_images  # noqa: E402
from lowlight.dl.infer import DLEnhancer, load_model  # noqa: E402
from lowlight.hybrid import HybridConfig, HybridEnhancer  # noqa: E402
from lowlight.quality import analyze  # noqa: E402
from lowlight.utils import load_image, resize_max_side  # noqa: E402
from lowlight.viz import NICE_NAMES, to_display  # noqa: E402

OUT = ROOT / "report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
model, ok = load_model()
assert ok, "weights/zero_dce.pt missing"
dl = DLEnhancer(model=model)
ENH = {"Classical": ClassicalEnhancer(ClassicalConfig()), "Deep learning": dl, "Hybrid": HybridEnhancer(dl, HybridConfig())}


def run(img):
    return {k: e.enhance(img) for k, e in ENH.items()}


def grid(rows, titles, path, caption_fn=None, width=2.3):
    """rows: list of lists of images (same columns)."""
    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(width * nc, width * 0.72 * nr + 0.35), squeeze=False)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            ax = axes[r][c]
            ax.imshow(np.clip(im, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=10.5, pad=3)
            if caption_fn:
                t = caption_fn(r, c)
                if t:
                    ax.set_xlabel(t, fontsize=8.6, labelpad=2)
    fig.subplots_adjust(wspace=0.03, hspace=0.12, left=0.005, right=0.995, top=0.94, bottom=0.01)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"quality": 86})
    plt.close(fig)
    print("saved", path.name)


# ---- 1. extreme low light (paired: input | classical | deep | hybrid | ground truth) ----
lol = {p[0].stem: p for p in find_pairs(ROOT / "data/LOL/eval15")}
v2 = find_pairs(ROOT / "data/LOL-v2/Real_captured/Test")
cands = [(lol["55"], "LOL 55")]
# darkest LOL-v2 Real test image
darkest = min(v2, key=lambda p: analyze(resize_max_side(load_image(p[0]), 300)).mean_lum)
cands.append((darkest, "LOL-v2 " + darkest[0].stem))
cands.append((lol["146"], "LOL 146"))
rows, info = [], []
for (lp, hp), name in cands:
    x = resize_max_side(load_image(lp), 600); gt = resize_max_side(load_image(hp), 600)
    res = run(x)
    rows.append([x, res["Classical"].image, res["Deep learning"].image, res["Hybrid"].image, gt])
    ps = {k: M.psnr(gt, v.image) for k, v in res.items()}
    ss = {k: M.ssim(gt, v.image) for k, v in res.items()}
    info.append((name, analyze(x).mean_lum, ps, ss))
grid(rows, ["Raw input", "Classical", "Deep learning (Zero-DCE)", "Hybrid (proposed)", "Ground truth"],
     OUT / "fig1_extreme_lowlight.jpg",
     lambda r, c: (f"{info[r][0]} (luma {info[r][1]:.2f})" if c == 0 else
                   (f"PSNR {list(info[r][2].values())[c - 1]:.1f} dB  SSIM {list(info[r][3].values())[c - 1]:.2f}" if 1 <= c <= 3 else "")))

# ---- 2. non-uniform / backlit scenes (unseen, no ground truth) ----
unseen = []
for s in ("DICM", "LIME", "MEF", "NPE", "VV", "Fusion"):
    unseen += list_images(ROOT / "data/unseen_src/Test" / s)
scored = []
for p in unseen:
    x = resize_max_side(load_image(p), 300); st = analyze(x)
    # backlit / mixed lighting: a lot of clipped-bright area AND a large dark fraction AND high non-uniformity
    scored.append((st.non_uniformity * min(st.saturated_frac * 20, 1) * st.dark_frac, p))
scored.sort(key=lambda t: -t[0])
picked = []
for _, p in scored:
    if all(p.parent != q.parent for q in picked):
        picked.append(p)
    if len(picked) == 3:
        break
rows, over = [], []
for p in picked:
    x = resize_max_side(load_image(p), 600); res = run(x)
    rows.append([x, res["Classical"].image, res["Deep learning"].image, res["Hybrid"].image])
    over.append([M.overexposure(x)] + [M.overexposure(res[k].image) for k in ENH])
grid(rows, ["Raw input", "Classical", "Deep learning (Zero-DCE)", "Hybrid (proposed)"], OUT / "fig2_nonuniform_backlit.jpg",
     lambda r, c: f"over-exposed {over[r][c] * 100:.1f}%", width=2.4)
print("backlit picks:", [f"{p.parent.name}/{p.name}" for p in picked])

# ---- 3. intermediate representations (hybrid on LOL 146) ----
(lp, hp) = lol["146"]
x = resize_max_side(load_image(lp), 600)
r = ENH["Hybrid"].enhance(x)
keys = ["illumination", "gain", "reflectance", "classical_branch", "dl_branch", "w_dl", "final"]
imgs = [np.clip(x * 3, 0, 1)] + [to_display(k, r.stages[k]) for k in keys]
SHORT = {"illumination": "Illumination map L", "gain": "Illumination gain map (heat)", "reflectance": "Reflectance R = I / L",
         "classical_branch": "Classical branch output", "dl_branch": "Deep branch output", "w_dl": "Deep-branch fusion weight (heat)", "final": "Final hybrid output"}
names = ["Input (x3 brightened for display)"] + [SHORT[k] for k in keys]
fig, axes = plt.subplots(2, 4, figsize=(11, 4.9))
for ax, im, n in zip(axes.ravel(), imgs, names):
    ax.imshow(np.clip(im, 0, 1)); ax.set_title(n, fontsize=9.5, pad=3); ax.axis("off")
fig.subplots_adjust(wspace=0.03, hspace=0.12, left=0.005, right=0.995, top=0.95, bottom=0.01)
fig.savefig(OUT / "fig3_intermediate.jpg", dpi=150, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"quality": 86}); plt.close(fig)
print("saved fig3_intermediate.jpg")
