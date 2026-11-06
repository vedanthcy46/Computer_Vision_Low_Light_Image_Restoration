# Design document - P1: Robust Low-Light Image Restoration and Enhancement

## 1. Problem and requirements

Restore images captured in very dark, non-uniformly lit, noisy and colour-cast conditions *without* over-exposure, halos or
colour shifts, preserving edges and texture. The brief requires a classical pipeline, a deep-learning model, a hybrid of the two,
an adaptive (non-fixed) strategy, a three-way comparison, an ablation study, over-exposure / colour-shift / detail-loss analysis,
testing on unseen images, PSNR/SSIM/LPIPS/NIQE/latency/noise/edge metrics and a deployed application.

| Requirement (brief) | Where it is met |
|---|---|
| Illumination estimation & decomposition, Retinex | `retinex.py` (guided max-RGB decomposition `I = R·L`; MSRCP variant) |
| CLAHE | `classical.apply_clahe` (16-bit Lab lightness) |
| Bilateral / guided edge-preserving denoise | `filters.denoise` (guided with per-pixel eps, or bilateral; chroma smoothed more) |
| Contrast and colour correction | tone stretch, CLAHE, grey-world + saturation (`classical.py`) |
| Adaptive parameter selection | `quality.py` -> `adaptive.select_params` |
| RetinexNet / Zero-DCE / EnlightenGAN | **Zero-DCE** (`dl/zero_dce.py`) |
| Fine-tune on available data | `scripts/train.py` (zero-reference, supervised, or mixed) on LOL / LOL-v2 / FiveK |
| Generalisation, image-quality-aware strategy | `dl/infer.DLEnhancer(quality_aware=True)`; evaluation on LOL-v2 + unseen folder |
| Classical + DL integration, adaptive combination | `hybrid.py` (illumination-gated pyramid fusion) |
| Compare classical / DL / hybrid; ablation | `scripts/evaluate.py`, `variants.py` |
| Over-exposure, colour shift, detail loss | `metrics.overexposure / colour_shift / detail_ratio`, output guard |
| Deployed app, intermediate visualisations, failure analysis | `app/streamlit_app.py`, `failure_analysis.md` |

## 2. Data

LOL (485 train / 15 test pairs), LOL-v2 (Real_captured and Synthetic, train/test), MIT-Adobe FiveK (well-exposed images for
NIQE fitting / synthetic degradation) and Zero-DCE's recipe (see `data/README.md`). Splits: fine-tune only on `Train`; all reported
numbers use held-out `Test` splits and an `unseen/` folder evaluated with no-reference metrics only.
`lowlight/data.py::degrade_low_light` provides a physically motivated synthetic degradation (linear-light attenuation with a
non-uniform field and a light-source blob, per-channel colour cast, signal-dependent shot + read noise, 8-bit quantisation) used for
tests, smoke runs and optional augmentation.

## 3. Method

**3.1 Image-quality analysis (`quality.py`).** Mean/percentile luma, dark-pixel fraction, robust (MAD-Immerkaer) noise sigma,
non-uniformity (dynamic range of the blurred log-luma), colour cast (`max|channel mean / grey mean - 1|`) and a 0-1 darkness score.

**3.2 Adaptive parameters (`adaptive.py`).** All parameters are smooth functions of the statistics, e.g. illumination exponent
`gamma = log(target)/log(mean max-RGB)`, gain cap `5 + 11(1 - noise)`, denoise threshold `k = 1.2 + 2 noise`, CLAHE clip
`1.2 + 2.8 darkness (1 - 0.6 noise)`, overall `strength ≈ max(1.25 darkness, dark fraction)` (well-exposed inputs are left alone),
deep-branch prior `0.8 - 0.5 noise`.

**3.3 Classical pipeline.** (1) `L = max(guidedfilter(max-RGB), max-RGB)` so that `R = I/L <= 1`. (2) Brighten with the illumination
curve `L' = L^gamma`, i.e. a per-pixel gain `G = L^(gamma-1)` capped at `gain_max`. (3) **Gain-aware denoising**: amplified noise has
std `sigma·G`, so the guided filter's `eps` is the per-pixel map `(k·sigma·G)^2` (+ stronger chroma smoothing). (4) **Wiener detail
shrinkage**: base/detail split; detail is attenuated by `max(E - v, 0)/E` where `E` is local detail energy and `v` the residual noise
variance, so strong edges/texture pass and noise is removed. (5) Percentile tone stretch (Retinex gain flattens contrast), CLAHE on
Lab lightness, partial grey-world white balance + chroma gain, soft `tanh` highlight roll-off, strength blend.

**3.4 Deep learning.** Zero-DCE: a 7-layer CNN predicts 8 per-pixel curve maps `LE(x) = x + a x (1-x)`. Zero-reference losses
(spatial consistency, exposure 0.6, colour constancy, curve smoothness) allow training without ground truth; a supervised L1 + SSIM
loss fine-tunes on paired data. *Quality-aware inference*: curve iterations scale with darkness, noisy inputs are lightly denoised first,
highlights are rolled off, an exposure-feedback step pulls back over-bright results and the strength follows the measured darkness.

**3.5 Hybrid.** Both branches enhance the same input. Candidate weights `w = contrast^0.6 · saturation^0.4 · wellExposed` (contrast is
measured on a smoothed luma so noise is not rewarded) are multiplied by an **illumination prior**: `p_dl = dl_bias · (0.4 + 0.6
smoothstep(L, 0.03, 0.2))`, i.e. very dark regions rely more on the explicitly noise-controlled classical branch while better-lit regions
use the learned curves. Weights are guided-filter smoothed and candidates are blended in a **Laplacian pyramid** (no halos/seams),
followed by the output guard.

## 4. Evaluation protocol

`scripts/evaluate.py` runs every method on every image (warm-up, then timed), computing PSNR, SSIM, LPIPS, NIQE, latency, noise
suppression, edge preservation, over-exposure, chroma shift and detail ratio, writes the `eval_P1_<TEAM>.md` deliverable (first line =
USNs), an ablation over 26 variants (`variants.py`) and an automatic failure analysis (worst images, annotated with measured failure modes).
Baselines for reference: identity, global auto-gamma, global histogram equalisation.

## 5. Results

> **Status of the numbers.** The dataset hosts (LOL, Kaggle, FiveK) were unreachable from the sandbox in which this code was developed,
> so the table below is a **synthetic-data smoke run** (16 degraded images, 600 px max; Zero-DCE weights trained for 40 epochs on only
> 7 synthetic images, LPIPS/NIQE backbones not downloadable -> n/a). It demonstrates the evaluation machinery and the *relative* contribution
> of the classical components; it is **not** a result on LOL. The deep-learning and hybrid rows in particular will change once
> `weights/zero_dce.pt` is trained on LOL (README -> Reproducing) and `scripts/evaluate.py` is re-run; replace this section with those numbers.

| Method (synthetic smoke run) | PSNR ↑ | SSIM ↑ | Latency ms ↓ (256 px) | Noise suppr. ↑ | Edge pres. ↑ | Chroma dE ↓ |
|---|---|---|---|---|---|---|
| input | 9.86 | 0.284 | 0 | 1.0 | 0.605 | 21.4 |
| auto-gamma | 14.50 | 0.281 | 1.4 | 1.3 | 0.406 | 27.9 |
| histogram eq. | 12.23 | 0.292 | 1.3 | 1.0 | 0.545 | 22.0 |
| **classical** | 16.39 | 0.586 | 34.7 | 8.4 | 0.590 | 11.4 |
| deep learning (smoke-trained) | 11.29 | 0.404 | 61.2 | 4.4 | 0.492 | 34.2 |
| **hybrid** | 15.97 | 0.568 | 106.8 | 9.2 | 0.535 | 17.6 |

Observations that are likely to transfer: both classical and hybrid beat naive global methods by a wide margin and suppress noise
8-9x; the under-trained deep branch is the weakest and drags the hybrid slightly below the classical pipeline *in this run* - the hybrid is
only as good as its learned branch, which is why training on real data matters.

### Ablation (synthetic smoke run, PSNR dB / SSIM / noise suppression)

| Removed component | classical | hybrid |
|---|---|---|
| none (full) | 16.39 / 0.586 / 8.4 | 15.97 / 0.568 / 9.2 |
| gain-aware denoising | 13.97 / 0.250 / 0.6 (**-2.4 dB**) | 14.24 / 0.267 / 0.7 (-1.7 dB) |
| Wiener shrinkage | 16.23 / 0.461 / 2.4 (SSIM -0.13) | 16.12 / 0.472 / 3.0 (SSIM -0.10) |
| CLAHE | 15.08 / 0.553 / 23.2 (-1.3 dB; CLAHE itself costs ~2.8x noise suppression) | 15.11 / 0.530 / 18.4 (-0.9 dB) |
| tone stretch | 15.93 / 0.591 / 9.6 (-0.5 dB) | 15.38 / 0.558 / 10.7 (-0.6 dB) |
| adaptive parameters (fixed instead) | **16.81** / 0.586 / 4.6 (+0.4 dB, noise x1.8 worse) | 14.95 / 0.525 / 5.4 (**-1.0 dB**) |
| colour correction | 16.69 / 0.591 / 8.7 (+0.3 dB and lower chroma dE with it *off*) | 16.09 / 0.569 / 9.4 |
| illumination prior (fusion) | - | 15.00 / 0.543 / 10.4 (-1.0 dB) |
| pyramid blending (alpha instead) | - | 15.85 / 0.561 / 9.9 (-0.1 dB) |
| DL pre-denoise | - | 13.40 / 0.285 / 1.3 (**-2.6 dB**) |
| deep branch (= classical only) | - | 16.39 / 0.586 (+0.4 dB, untrained DL) |
| quality-aware strategy (DL) | 11.29 / 0.404 (with) vs 9.29 / 0.168 (raw Zero-DCE) | |
| MSRCP instead of illumination decomposition | 14.32 / 0.446 (-2.1 dB) | |

Reading: denoising/shrinkage are the load-bearing components; the quality-aware wrapper more than doubles raw Zero-DCE's SSIM; the
illumination prior and adaptive parameters help the *hybrid* by ~1 dB; on this synthetic set fixed parameters score slightly higher PSNR
for the classical pipeline but with 1.8x more residual noise, and removing colour correction improves PSNR/SSIM a little (synthetic images have
a mild random cast and a neutral ground truth) - re-check on LOL before changing defaults.

## 6. Failure-case analysis

`results/<run>/failure_analysis.md` lists the worst hybrid outputs with their measured input diagnostics and automatically detected
failure modes. Failure modes **1 and 4 were observed** in the synthetic run; **2 and 3 are expected by construction but have not yet been
demonstrated** - confirm or refute them on LOL / unseen images and update this section with the real cases:

1. **Extremely dark + very noisy input (noise sigma > 0.025, darkness > 0.95)** - the signal is below the noise floor, so the pipeline
   must choose between noise and detail. It removes noise (suppression x10-15) at the cost of smeared fine texture (detail ratio ~0.5,
   e.g. fur and hair in `syn05`, `syn08`). Cause: the gain-aware denoiser's threshold is `k·sigma·G` and gains reach the cap (up to 16x).
2. **Strongly non-uniform lighting with a localised light source** - the illumination estimate blends the bright blob into surrounding
   shadows, leaving soft halos around the source or under-lit shadows that CLAHE then over-amplifies. Cause: a single smoothing radius
   cannot match both the source boundary and the shadow structure.
3. **Colour cast under coloured illumination** - grey-world correction assumes a neutral scene; for a dominant legitimate colour
   (neon, sunset) it desaturates or shifts hue. Cause: the cast estimate cannot distinguish illuminant colour from scene colour; mitigated
   by capping the correction strength at 0.6 and the gain range to [0.75, 1.33].
4. **Under-trained / domain-shifted deep branch** - raw Zero-DCE desaturates and washes out dark, noisy inputs (grey/blue-white cast)
   because its zero-reference exposure loss ignores noise and colour fidelity; the quality-aware wrapper and the hybrid weights reduce, but
   cannot remove it. Fine-tuning on paired LOL data is the intended remedy.

## 7. Deployment and UX

Streamlit app (`app/streamlit_app.py`): upload (PNG/JPEG/TIFF/BMP/WebP, 25 MB cap, EXIF-aware, auto down-scaling), method selector
(hybrid / classical / deep / compare all), optional reference image (enables PSNR/SSIM/LPIPS), synthetic-sample generator for demos,
tabs for result + download, intermediate pipeline steps (illumination, reflectance, gain heat-map, denoise, shrinkage, CLAHE, colour,
fusion weights), measured statistics + chosen parameters + histograms, and metrics. Corrupt files, oversized files and processing errors
produce readable messages instead of stack traces; the model is cached across requests. Measured on a 4-core CPU at 800x600: classical ~0.25 s, deep ~0.65 s, hybrid ~1.1 s. Container image via `Dockerfile` (CPU PyTorch), no secrets.

## 8. Limitations, ethics, future work

See README. Future work: learned denoiser in front of the curves (e.g. a small U-Net trained on LOL), per-region radius for illumination
estimation, learned fusion weights, int8/fp16 export of the 79k-parameter network for mobile, RAW-domain processing.
