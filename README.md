---
title: Low-Light Image Restoration
emoji: 🌙
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# Robust Low-Light Image Restoration and Enhancement (Problem Statement P1)

Illumination-adaptive restoration of severely under-exposed, noisy, colour-distorted images that preserves edges and
texture and avoids over-exposure, halos and colour shifts. Three systems are implemented and compared:

| | Approach | Core ideas |
|---|---|---|
| **Classical** | `lowlight/classical.py` | Retinex illumination decomposition (guided max-RGB), gain-capped illumination curve, gain-aware guided/bilateral denoising, Wiener detail shrinkage, tone stretch, CLAHE, grey-world colour correction, soft highlight roll-off |
| **Deep learning** | `lowlight/dl/` | Zero-DCE (79,416 params) re-implemented in PyTorch; zero-reference training + supervised fine-tuning on paired data; **image-quality-aware** strength / iterations / exposure feedback |
| **Hybrid** | `lowlight/hybrid.py` | Classical illumination map gates a Laplacian-pyramid fusion of the classical and deep branches using well-exposedness, denoised contrast, saturation and an illumination prior |

Everything is driven by `lowlight/quality.py` + `lowlight/adaptive.py`: exposure, noise level, lighting non-uniformity and
colour cast are measured per image and mapped to the parameters of every stage - **no fixed enhancement constants**.
Already well-exposed images are left almost untouched (no over-exposure).

Architecture: [`docs/architecture.md`](docs/architecture.md) · Design document: [`docs/design.md`](docs/design.md)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU PyTorch is enough
python -m pytest -q                        # 38 tests (needs: pip install -r requirements-dev.txt)
streamlit run app/streamlit_app.py         # http://localhost:8501
# or:  docker build -t lowlight . && docker run -p 8501:8501 lowlight
```

Without `weights/zero_dce.pt` the app only offers the classical pipeline (see *Training*).

## Reproducing the results

1. **Get the datasets** (not in the repo) - see [`data/README.md`](data/README.md): LOL, LOL-v2 (Kaggle) and optionally MIT-Adobe FiveK.
2. **Train / fine-tune Zero-DCE** (CPU is fine: tiny network; a GPU is used automatically if present)

   ```bash
   # a) zero-reference training on the low-light train images (no ground truth used)
   python scripts/train.py --low data/LOL/our485/low data/LOL-v2/Real_captured/Train/Low --epochs 60
   # b) supervised fine-tuning on the paired train splits (L1 + SSIM loss)
   python scripts/train.py --pairs data/LOL/our485 data/LOL-v2/Real_captured/Train \
          --init weights/zero_dce.pt --epochs 30 --lr 5e-5
   git add weights/zero_dce.pt            # ~330 KB, needed by the deployed app
   ```
3. **Fit the NIQE reference model** on well-exposed images (optional, offline NIQE):
   `python scripts/fit_niqe.py data/LOL/our485/high data/LOL-v2/Real_captured/Train/Normal`
4. **Evaluate** (comparison + ablation + failure analysis + unseen images):

   ```bash
   python scripts/evaluate.py --data data/LOL/eval15 data/LOL-v2/Real_captured/Test data/LOL-v2/Synthetic/Test \
          --unpaired data/unseen --ablation --team T01 --usns USN1,USN2,USN3 --out results/final
   cp results/final/eval_P1_T01.md .          # hackathon deliverable (first line = USNs)
   ```

   Smoke test without any download: `python scripts/evaluate.py --synthetic 12 --allow-untrained --ablation`.

Outputs: `per_image.csv`, `summary.csv`, `ablation.csv`, `visuals/`, `failure_analysis.md`, `eval_P1_<TEAM>.md`.

## Metrics

PSNR, SSIM (reference), **LPIPS** (AlexNet, `lpips` package), **NIQE** (official parameters via `pyiqa`, or the self-fitted local
model), processing latency, **noise suppression** (noise of the equally-bright naive input / noise of the output),
**edge preservation** (edge-map correlation), plus over-exposure fraction, chroma shift (dE on a,b) and detail ratio.
LPIPS and `pyiqa` need to download pretrained backbones on first use (internet); when unavailable they are reported as `n/a`
rather than invented.

## Repository layout

```
lowlight/   quality.py adaptive.py retinex.py filters.py classical.py hybrid.py variants.py metrics.py niqe.py data.py viz.py api.py
lowlight/dl/  zero_dce.py losses.py infer.py
scripts/    train.py evaluate.py fit_niqe.py
app/        streamlit_app.py        tests/   docs/   data/   weights/   Dockerfile
```

## Deployment

The app is a single Streamlit process (CPU: ~0.25 s classical / ~1.1 s hybrid for an 800x600 image on 4 cores). Any container host works with the provided
`Dockerfile` (it honours `$PORT`): Hugging Face Spaces (Docker SDK, set `app_port: 8501`), Render, Railway, Fly.io, Cloud Run.
Streamlit Community Cloud also works directly from `requirements.txt` + `app/streamlit_app.py`. No secrets are needed.

## Limitations and ethics

* Quality on real photographs depends on the trained `zero_dce.pt`; the checked-in code was verified on synthetic degradations
  only, because the dataset hosts were unreachable in the build sandbox. Run the *Reproducing* steps to obtain real numbers.
* Extremely dark, noisy pixels contain no recoverable detail; the system trades detail for noise suppression there
  (see failure analysis). Enhanced regions are *plausible reconstructions*, not ground truth - do not use for forensic,
  medical or evidentiary purposes, and disclose enhancement when sharing images.
* Colour correction assumes a roughly neutral scene; strongly coloured lighting (neon) may be partly "corrected".
