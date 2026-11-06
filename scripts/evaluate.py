"""Comparative evaluation, ablation study and failure-case analysis.

Examples
--------
LOL + LOL-v2 test splits (paired -> PSNR/SSIM/LPIPS) and an unpaired folder of unseen images::

    python scripts/evaluate.py --data data/LOL/eval15 data/LOL-v2/Real_captured/Test \\
        --unpaired data/unseen --team T01 --usns 1XX23CS001,1XX23CS002 --ablation

Smoke run without downloads (synthetic degradations of bundled images)::

    python scripts/evaluate.py --synthetic 12 --allow-untrained --ablation

Outputs (in ``--out``, default results/<tag>/):
  per_image.csv, summary.csv, ablation.csv, visuals/*.png, failure_analysis.md and the
  hackathon deliverable ``eval_P1_<TEAM>.md`` (first line: comma separated USNs).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight import metrics as M  # noqa: E402
from lowlight.data import find_pairs, list_images, make_synthetic_pairs  # noqa: E402
from lowlight.dl.infer import DLEnhancer, load_model  # noqa: E402
from lowlight.quality import analyze  # noqa: E402
from lowlight.utils import load_image, resize_max_side, save_image, side_by_side  # noqa: E402
from lowlight.variants import ablation_variants, main_methods  # noqa: E402

PRIMARY = ["input", "gamma", "hist_eq", "classical", "deep_learning", "hybrid"]


def load_datasets(a) -> dict[str, list[tuple[str, np.ndarray, np.ndarray | None]]]:
    ds: dict[str, list] = {}
    for root in a.data or []:
        pairs = find_pairs(root)
        if not pairs:
            print(f"WARNING: no low/high pairs found under {root}")
            continue
        if a.max_images:
            pairs = pairs[: a.max_images]
        ds[Path(root).as_posix().replace("/", "_")] = [
            (lp.stem, resize_max_side(load_image(lp), a.size), resize_max_side(load_image(hp), a.size))
            for lp, hp in pairs
        ]
    for d in a.unpaired or []:
        paths = list_images(d)[: a.max_images or None]
        if paths:
            ds["unseen_" + Path(d).name] = [(p.stem, resize_max_side(load_image(p), a.size), None) for p in paths]
    if a.synthetic:
        ds["synthetic"] = [(f"syn{i:02d}", lo, gt) for i, (lo, gt) in enumerate(make_synthetic_pairs(a.synthetic, 256, a.seed))]
    return ds


def run_methods(methods: dict, data, niqe_backend: str, desc: str, keep_outputs: bool = False):
    """Run every method on every image. Returns per-image DataFrame (+ cached outputs)."""
    rows, outputs = [], {}
    for name, fn in methods.items():
        fn(data[0][1])  # warm-up (model load, allocator, caches)
        for iid, lo, gt in data:
            t = time.perf_counter()
            out = fn(lo)
            ms = (time.perf_counter() - t) * 1000.0
            out = np.asarray(getattr(out, "image", out), dtype=np.float32)
            r = M.evaluate_pair(lo, out, gt, ms, niqe_backend)
            r.update(method=name, image=iid)
            rows.append(r)
            if keep_outputs:
                outputs[(name, iid)] = out
        print(f"  [{desc}] {name:<55s} done", flush=True)
    return pd.DataFrame(rows), outputs


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in M.METRIC_ORDER + M.EXTRA_ORDER if c in df]
    return df.groupby("method", sort=False)[cols].mean(numeric_only=True)


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append("n/a" if pd.isna(v) else (f"{v:.1f}" if c == "Latency_ms" else floatfmt.format(v)))
        out.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(out)


ARROWS = {"PSNR": "↑", "SSIM": "↑", "LPIPS": "↓", "NIQE": "↓", "Latency_ms": "↓ (ms)", "NoiseSuppression": "↑",
          "EdgePreservation": "↑", "Overexposure": "↓", "ColourShift_dE": "↓", "DetailRatio": "≈1", "MeanLuma": ""}


def failure_tags(row, stats) -> list[str]:
    tags = []
    if row["MeanLuma"] < 0.22:
        tags.append("under-enhanced (output still dark)")
    if row["MeanLuma"] > 0.7 or row["Overexposure"] > 0.02:
        tags.append("over-exposure")
    if not pd.isna(row["ColourShift_dE"]) and row["ColourShift_dE"] > 18:
        tags.append("colour shift")
    if row["DetailRatio"] < 0.55:
        tags.append("detail loss / over-smoothing")
    if row["DetailRatio"] > 1.8 or row["NoiseSuppression"] < 1.5:
        tags.append("residual / amplified noise")
    if row["EdgePreservation"] < 0.3:
        tags.append("edges not preserved")
    if stats.non_uniformity > 0.6:
        tags.append("strongly non-uniform lighting in input")
    if stats.noise_level > 0.7:
        tags.append("very noisy input")
    if stats.darkness > 0.97:
        tags.append("extremely dark input")
    return tags or ["no dominant failure mode (just lowest score)"]


def failure_analysis(df, datasets, outputs, outdir: Path, n: int) -> str:
    lines = ["# Failure-case analysis (hybrid method)\n"]
    h = df[df.method == "hybrid"].copy()
    key = "PSNR" if h["PSNR"].notna().any() else "NIQE"
    # worst = lowest PSNR, or highest NIQE (lower NIQE is better)
    h = h.sort_values(key, ascending=(key == "PSNR"))
    lines.append(f"Worst {n} images by **{key}** ({'lowest' if key == 'PSNR' else 'highest'} = worst). "
                 "Failure tags are derived automatically from measured diagnostics; the written causes in "
                 "`docs/design.md` summarise them.\n")
    lookup = {iid: (lo, gt) for data in datasets.values() for iid, lo, gt in data}
    (outdir / "visuals").mkdir(parents=True, exist_ok=True)
    for _, r in h.head(n).iterrows():
        lo, gt = lookup[r["image"]]
        st = analyze(lo)
        tags = failure_tags(r, st)
        panels = [lo, np.clip(lo * 4, 0, 1)] + [outputs[(m, r["image"])] for m in ("classical", "deep_learning", "hybrid")]
        if gt is not None:
            panels.append(gt)
        save_image(outdir / "visuals" / f"fail_{r['image']}.png", side_by_side(*panels))
        lines += [
            f"## {r['image']}",
            f"![{r['image']}](visuals/fail_{r['image']}.png)",
            "",
            "*Panels: input | input x4 (for visibility) | classical | deep learning | hybrid" + (" | ground truth*" if gt is not None else "*"),
            "",
            f"- PSNR {r['PSNR']:.2f} dB, SSIM {r['SSIM']:.3f}, NIQE {r['NIQE']:.2f}, noise suppression x{r['NoiseSuppression']:.1f}, "
            f"edge preservation {r['EdgePreservation']:.2f}, mean output luma {r['MeanLuma']:.2f}"
            if not pd.isna(r["PSNR"]) else f"- NIQE {r['NIQE']:.2f}, noise suppression x{r['NoiseSuppression']:.1f}, mean output luma {r['MeanLuma']:.2f}",
            f"- Input diagnostics: mean luma {st.mean_lum:.3f}, noise sigma {st.noise_sigma:.4f}, non-uniformity {st.non_uniformity:.2f}, colour cast {st.color_cast:.2f}",
            f"- **Detected failure modes:** {', '.join(tags)}",
            "",
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="*", help="paired dataset roots (folders containing low/ + high/ or Low/ + Normal/)")
    ap.add_argument("--unpaired", nargs="*", help="folders of unseen low-light images without ground truth")
    ap.add_argument("--synthetic", type=int, default=0, help="add N synthetic pairs (smoke test)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--allow-untrained", action="store_true", help="run even if no Zero-DCE weights exist (meaningless DL numbers)")
    ap.add_argument("--team", default="T01")
    ap.add_argument("--usns", default="USN1,USN2", help="comma separated USNs (first line of the eval file)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=600, help="max image side during evaluation")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--niqe", default="auto", choices=["auto", "pyiqa", "local"])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--failures", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model, loaded = load_model(a.weights)
    if not loaded and not a.allow_untrained:
        sys.exit("No Zero-DCE weights found (weights/zero_dce.pt). Train with scripts/train.py or pass --weights; "
                 "use --allow-untrained only for smoke tests.")
    dl = DLEnhancer(model=model)
    datasets = load_datasets(a)
    if not datasets:
        sys.exit("No data. Use --data / --unpaired / --synthetic N.")
    outdir = Path(a.out or f"results/{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"LPIPS available: {M.lpips_available()} | NIQE backend: {a.niqe} | output: {outdir}")

    methods = main_methods(dl)
    summaries, all_df, all_out = {}, [], {}
    for dname, data in datasets.items():
        print(f"\n== dataset {dname} ({len(data)} images)")
        df, outs = run_methods(methods, data, a.niqe, dname, keep_outputs=True)
        df["dataset"] = dname
        all_df.append(df)
        all_out.update(outs)
        summaries[dname] = summarise(df)
        (outdir / "visuals").mkdir(exist_ok=True)
        for iid, lo, gt in data[: min(4, len(data))]:
            pan = [lo, np.clip(lo * 4, 0, 1)] + [outs[(m, iid)] for m in ("classical", "deep_learning", "hybrid")] + ([gt] if gt is not None else [])
            save_image(outdir / "visuals" / f"cmp_{dname}_{iid}.png", side_by_side(*pan))
    full = pd.concat(all_df)
    full.to_csv(outdir / "per_image.csv", index=False)
    pd.concat(summaries, names=["dataset"]).to_csv(outdir / "summary.csv")

    abl_md = ""
    if a.ablation:
        print("\n== ablation")
        variants = ablation_variants(dl)
        paired = {k: v for k, v in datasets.items() if all(g is not None for _, _, g in v)}
        pool = paired or datasets
        adata = [item for v in pool.values() for item in v][: (a.max_images or 40)]
        adf, _ = run_methods(variants, adata, a.niqe, "ablation")
        asum = summarise(adf)
        asum.to_csv(outdir / "ablation.csv")
        cols = [c for c in ["PSNR", "SSIM", "LPIPS", "NIQE", "Latency_ms", "NoiseSuppression", "EdgePreservation", "Overexposure", "ColourShift_dE"] if c in asum]
        t = asum[cols].copy()
        for ref_name, prefix in (("hybrid (full)", "hybrid"), ("classical (full)", "classical")):
            if ref_name in t.index and "PSNR" in t:
                t[f"dPSNR vs {prefix}"] = np.where(t.index.str.startswith(prefix), t["PSNR"] - t.loc[ref_name, "PSNR"], np.nan)
        abl_md = ("\n## Ablation study\n\nEvery row removes (or swaps) exactly one component of the full system; "
                  "the `dPSNR` columns give the change relative to the full pipeline of the same family "
                  f"(negative = the component helps). Evaluated on {len(adata)} images"
                  f"{' with ground truth' if paired else ' (no ground truth: PSNR/SSIM n/a)'}.\n\n" + md_table(t))

    fail_md = ""
    if any(m for m in all_out if m[0] == "hybrid"):
        fail_md = failure_analysis(full, datasets, all_out, outdir, a.failures)
        (outdir / "failure_analysis.md").write_text(fail_md, encoding="utf-8")

    # ------------------------------------------------------------------ eval_P1_<team>.md
    show = ["PSNR", "SSIM", "LPIPS", "NIQE", "Latency_ms", "NoiseSuppression", "EdgePreservation"]
    lines = [a.usns, "", f"# eval_P1_{a.team} - Robust Low-Light Image Restoration and Enhancement", "",
             "Metric order: " + ", ".join(f"{m} {ARROWS[m]}" for m in show) + ".",
             f"Run date {time.strftime('%Y-%m-%d')}; DL weights: {'trained checkpoint' if loaded else 'UNTRAINED (smoke test only)'}; "
             f"LPIPS {'AlexNet' if M.lpips_available() else 'NOT AVAILABLE (backbone could not be loaded)'}; NIQE backend `{a.niqe}`.", ""]
    for dname, s in summaries.items():
        n_img = len(datasets[dname])
        lines += [f"## {dname} ({n_img} images)", "", md_table(s[[c for c in show if c in s]]), "",
                  "Additional diagnostics:", "", md_table(s[[c for c in M.EXTRA_ORDER if c in s]]), ""]
    lines += [abl_md, "", "## Failure-case analysis", "", "See `failure_analysis.md` / `docs/design.md`.", ""]
    (outdir / f"eval_P1_{a.team}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {outdir / f'eval_P1_{a.team}.md'}")
    for dname, s in summaries.items():
        print(f"\n{dname}\n{s.round(3).to_string()}")


if __name__ == "__main__":
    main()
