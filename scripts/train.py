"""Train / fine-tune Zero-DCE.

Examples
--------
Zero-reference training on LOL (+ LOL-v2) low-light images (no ground truth needed)::

    python scripts/train.py --low data/LOL/our485/low data/LOL-v2/Real_captured/Train/Low --epochs 60

Supervised fine-tuning on paired data (LOL / LOL-v2 train splits)::

    python scripts/train.py --pairs data/LOL/our485 data/LOL-v2/Real_captured/Train \\
        --init weights/zero_dce.pt --mode supervised --epochs 30 --lr 5e-5

Smoke run on built-in synthetic data (no downloads)::

    python scripts/train.py --synthetic --epochs 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight.data import PatchDataset, bundled_images, find_pairs, list_images  # noqa: E402
from lowlight.dl.infer import DEFAULT_WEIGHTS  # noqa: E402
from lowlight.dl.losses import SupervisedLoss, ZeroDCELoss  # noqa: E402
from lowlight.dl.zero_dce import DCENet, count_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", nargs="*", default=[], help="folders of low-light images (zero-reference training)")
    ap.add_argument("--pairs", nargs="*", default=[], help="dataset roots containing low/high pairs (supervised)")
    ap.add_argument("--synthetic", action="store_true", help="train on synthetic degradations of bundled images")
    ap.add_argument("--mode", choices=["zero_ref", "supervised", "mixed"], default=None,
                    help="default: zero_ref for --low/--synthetic, supervised for --pairs")
    ap.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--out", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction for validation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    low, gt = [], []
    for root in a.pairs:
        for lp, hp in find_pairs(root):
            low.append(lp), gt.append(hp)
    for d in a.low:
        low += list_images(d)
        gt += [None] * (len(low) - len(gt))
    mode = a.mode or ("supervised" if a.pairs and not (a.low or a.synthetic) else "zero_ref")

    if a.synthetic:
        ds_all = bundled_images(a.patch)
        n_val = max(1, int(len(ds_all) * a.val_frac))
        train_ds = PatchDataset([], None, a.patch, length_mult=16, synthetic_from=ds_all[:-n_val], seed=a.seed)
        val_ds = PatchDataset([], None, a.patch, length_mult=2, synthetic_from=ds_all[-n_val:], seed=a.seed + 1)
    else:
        if not low:
            ap.error("no training data: pass --low, --pairs or --synthetic")
        idx = np.random.default_rng(a.seed).permutation(len(low))
        n_val = max(1, int(len(low) * a.val_frac)) if len(low) > 1 else 0
        vi, ti = idx[:n_val], idx[n_val:]
        pick = lambda ids, L: [L[i] for i in ids]  # noqa: E731
        has_gt = all(g is not None for g in gt)
        train_ds = PatchDataset(pick(ti, low), pick(ti, gt) if has_gt else None, a.patch, length_mult=4, seed=a.seed)
        val_ds = PatchDataset(pick(vi, low), pick(vi, gt) if has_gt else None, a.patch, length_mult=1, seed=a.seed + 1)
        if mode != "zero_ref" and not has_gt:
            ap.error("supervised/mixed mode needs paired data (--pairs)")
    print(f"mode={mode} device={dev} train_items={len(train_ds)} val_items={len(val_ds)}")

    model = DCENet().to(dev)
    if a.init and Path(a.init).exists():
        ck = torch.load(a.init, map_location=dev, weights_only=True)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print("initialised from", a.init)
    print("parameters:", count_params(model))
    zr, sup = ZeroDCELoss().to(dev), SupervisedLoss().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    tl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=a.batch, num_workers=0)

    def loss_fn(lo, hi, enh, maps):
        if mode == "zero_ref":
            return zr(lo, enh, maps)[0]
        if mode == "supervised":
            return sup(lo, enh, maps, hi)[0]
        return zr(lo, enh, maps)[0] + sup(lo, enh, maps, hi)[0]

    best, out = float("inf"), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for lo, hi in tl:
            lo, hi = lo.to(dev), hi.to(dev)
            enh, maps = model(lo)
            loss = loss_fn(lo, hi, enh, maps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for lo, hi in vl:
                lo, hi = lo.to(dev), hi.to(dev)
                enh, maps = model(lo)
                vtot += loss_fn(lo, hi, enh, maps).item()
                vn += 1
        vloss = vtot / max(vn, 1)
        tag = ""
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "val_loss": vloss, "epoch": ep, "mode": mode}, out)
            tag = " *saved*"
        print(f"epoch {ep + 1:3d}/{a.epochs} train={tot / n:.4f} val={vloss:.4f} ({time.time() - t0:.1f}s){tag}")
    print("best val loss", best, "->", out)


if __name__ == "__main__":
    main()
