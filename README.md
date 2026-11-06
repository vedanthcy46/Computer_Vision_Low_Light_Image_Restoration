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
python -m pytest -q                        # 42 tests (needs: pip install -r requirements-dev.txt)
streamlit run app/streamlit_app.py         # http://localhost:8501
# or:  docker build -t lowlight . && docker run -p 8501:8501 lowlight
```

Without `weights/zero_dce.pt` the app only offers the classical pipeline (see *Training*).

## Reproducing the results

1. **Get the datasets** (not in the repo) - see [`data/README.md`](data/README.md): LOL, LOL-v2 (Kaggle), optionally the 1,000-pair BrighteningTrain set, and the unseen benchmark folders (DICM, LIME, MEF, NPE, VV, Fusion). We did not use MIT-Adobe FiveK.
2. **Train / fine-tune Zero-DCE** (CPU is fine: tiny network; a GPU is used automatically if present)

   ```bash
   # a) zero-reference training on the low-light train images (no ground truth used)
   python scripts/train.py --low data/LOL/our485/low data/LOL-v2/Real_captured/Train/Low data/LOL-v2/Synthetic/Train/Low data/BrighteningTrain/low --epochs 15 --out weights/zero_dce_zr.pt
   # b) supervised fine-tuning on the paired train splits (L1 + SSIM loss)
   python scripts/train.py --pairs data/LOL/our485 data/LOL-v2/Real_captured/Train data/LOL-v2/Synthetic/Train \
          --init weights/zero_dce_zr.pt --mode supervised --epochs 10 --lr 5e-5 --out weights/zero_dce.pt
   git add weights/zero_dce.pt weights/niqe_params.npz   # ~330 KB + 11 KB, needed by the deployed app
   ```
3. **Fit the NIQE reference model** on well-exposed images (optional, offline NIQE):
   `python scripts/fit_niqe.py data/LOL/our485/high data/LOL-v2/Real_captured/Train/Normal data/BrighteningTrain/high`
4. **Evaluate** (comparison + ablation + failure analysis + unseen images):

   ```bash
   python scripts/evaluate.py --data data/LOL/eval15 data/LOL-v2/Real_captured/Test data/LOL-v2/Synthetic/Test \
          --unpaired data/unseen/DICM data/unseen/LIME data/unseen/MEF data/unseen/NPE data/unseen/VV data/unseen/Fusion \
          --max-images 30 --ablation --failures 5 --team VisionX --usns 1RUA24CSE0525,1RUA24CSE0544 --out results/final
   cp results/final/eval_P1_VisionX.md .      # hackathon deliverable (first line = USNs)
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

**Live app: https://visionxlowlightrestoration.streamlit.app/** (Streamlit Community Cloud, CPU only, deployed from this repository's
`requirements.txt` + `app/streamlit_app.py`; free apps sleep when idle, so the first visit may take ~30 s to wake up).

The app is a single Streamlit process (CPU, <= 600 px: ~0.1-0.2 s classical, ~0.15-0.25 s deep, ~0.3-0.6 s hybrid on a laptop (~2 s for an 800 px hybrid run on the free Streamlit Cloud CPU); free Streamlit Cloud CPUs are slower). Any container host also works with the provided
`Dockerfile` (it honours `$PORT`): Hugging Face Spaces (Docker SDK; add the usual Space front-matter with `app_port: 8501` to the Space's README), Render, Railway, Fly.io, Cloud Run. No secrets are needed.

## Limitations and ethics

* Results (see `eval_P1_VisionX.md`, `docs/design.md` section 5) come from a **short** training run (15 epochs zero-reference + 10 epochs
  supervised on a laptop GPU) and at most 30 test images per set; they are not tuned and not a state-of-the-art claim. The hybrid wins
  SSIM/LPIPS/NIQE on LOL, but not PSNR everywhere (a plain gamma curve has higher PSNR on LOL-v2 Real, the deep branch on LOL-v2 Synthetic).
* NIQE is a locally fitted NIQE-style model (not the official parameters), so values are only comparable within this repository.
* Known weak spots (design doc section 6): dark saturated colours stay dull, very dark inputs keep some speckle and a slight colour tint,
  strongly non-uniform lighting under-lifts shadows, and PSNR on already-lit images is dominated by reference exposure.
* Extremely dark, noisy pixels contain no recoverable detail; the system trades detail for noise suppression there
  (see failure analysis). Enhanced regions are *plausible reconstructions*, not ground truth - do not use for forensic,
  medical or evidentiary purposes, and disclose enhancement when sharing images.
* Colour correction assumes a roughly neutral scene; strongly coloured lighting (neon) may be partly "corrected".
