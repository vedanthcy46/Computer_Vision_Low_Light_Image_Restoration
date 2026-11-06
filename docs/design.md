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

LOL (485 train / 15 test pairs), LOL-v2 (Real_captured 689/100 and Synthetic 900/100 train/test pairs), the 1,000-pair *BrighteningTrain* set (extra training images and well-exposed NIQE fitting data), the benchmark folders in `Test.zip` (DICM, LIME, MEF, NPE, VV, Fusion: used only as **unseen** images, never for training) and Zero-DCE's recipe (see `data/README.md`). MIT-Adobe FiveK was not used. Splits: fine-tune only on `Train`; all reported
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

All numbers below are produced by `scripts/evaluate.py` (full tables per dataset in `eval_P1_VisionX.md`, per-image values in
`results/final/per_image.csv`). Setup: Zero-DCE trained 15 epochs zero-reference (LOL + LOL-v2 + BrighteningTrain low-light images)
then fine-tuned 10 epochs supervised (L1 + SSIM) on the **train** splits of LOL, LOL-v2 Real and LOL-v2 Synthetic (7,468 patches);
**all reported numbers are on held-out test splits or on unseen images**. At most 30 images per set, resized to <= 600 px;
latency is single-image CPU inference. LPIPS is AlexNet-LPIPS. NIQE is a **locally fitted** NIQE-style model (300 well-exposed
LOL / LOL-v2 / Brightening images, `weights/niqe_params.npz`), so absolute NIQE values are not comparable with the official NIQE.
Best value per column (among the five methods, input excluded) is in bold. This is a short training schedule (25 epochs in total on a laptop GPU), not a tuned one.

### 5.1 Comparison: classical vs deep learning vs hybrid

**LOL eval15 (15 paired images)**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms ↓ | Noise suppr. ↑ | Edge pres. ↑ | Over-exp. ↓ | Colour dE ↓ |
|---|---|---|---|---|---|---|---|---|---|
| input (no enhancement) | 7.77 | 0.191 | 0.560 | 2.56 | 0 | 0.98 | 0.834 | 0.000 | 13.4 |
| global auto-gamma | 16.19 | 0.637 | 0.404 | 3.77 | 14 | 2.33 | 0.756 | **0.000** | 13.6 |
| global histogram eq. | 13.96 | 0.411 | 0.590 | 5.84 | 4 | 0.76 | 0.750 | 0.035 | 13.4 |
| **classical** | 17.05 | 0.663 | 0.328 | 3.21 | 126 | 1.89 | 0.848 | **0.000** | 10.0 |
| **deep learning** | **18.54** | 0.575 | 0.351 | 5.35 | 210 | 1.30 | 0.843 | **0.000** | 12.0 |
| **hybrid** | 18.16 | **0.716** | **0.284** | **3.01** | 436 | **2.39** | **0.870** | **0.000** | **9.9** |

**LOL-v2 Real_captured Test (30 paired images)**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms ↓ | Noise suppr. ↑ | Edge pres. ↑ | Over-exp. ↓ | Colour dE ↓ |
|---|---|---|---|---|---|---|---|---|---|
| input (no enhancement) | 7.93 | 0.182 | 0.557 | 2.61 | 0 | 0.98 | 0.757 | 0.000 | 7.3 |
| global auto-gamma | **18.86** | 0.660 | 0.392 | 3.93 | 8 | 2.19 | 0.754 | **0.000** | 9.6 |
| global histogram eq. | 15.98 | 0.462 | 0.520 | 5.74 | 3 | 0.81 | 0.766 | 0.025 | **7.3** |
| **classical** | 17.19 | 0.682 | 0.318 | 3.25 | 97 | 1.92 | 0.837 | **0.000** | 11.6 |
| **deep learning** | 17.99 | 0.549 | 0.388 | 5.43 | 167 | 1.19 | 0.816 | **0.000** | 12.9 |
| **hybrid** | 18.16 | **0.729** | **0.275** | **3.10** | 417 | **2.41** | **0.857** | **0.000** | 11.6 |

**LOL-v2 Synthetic Test (30 paired images)**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms ↓ | Noise suppr. ↑ | Edge pres. ↑ | Over-exp. ↓ | Colour dE ↓ |
|---|---|---|---|---|---|---|---|---|---|
| input (no enhancement) | 10.35 | 0.407 | 0.407 | 3.38 | 0 | 0.96 | 0.742 | 0.001 | 14.2 |
| global auto-gamma | 16.13 | 0.785 | 0.217 | 3.82 | 7 | **1.17** | 0.914 | 0.001 | 12.7 |
| global histogram eq. | 15.30 | 0.723 | 0.308 | 4.20 | 3 | 0.76 | 0.902 | 0.042 | 14.3 |
| **classical** | 14.90 | 0.717 | 0.257 | **3.46** | 95 | 0.82 | 0.903 | **0.000** | 9.7 |
| **deep learning** | **16.95** | **0.792** | **0.179** | 4.16 | 149 | 0.83 | **0.932** | **0.000** | **8.7** |
| **hybrid** | 16.31 | 0.769 | 0.215 | 3.67 | 334 | 0.95 | 0.920 | **0.000** | 8.8 |

**Unseen images without ground truth (DICM, LIME, MEF, NPE, VV, Fusion; mean of the six per-set means)**

| Method | NIQE ↓ | Noise suppr. ↑ | Edge pres. ↑ | Over-exposed px ↓ | Detail ratio | Mean luma | Latency ms |
|---|---|---|---|---|---|---|---|
| input (no enhancement) | 2.91 | 0.95 | 1.000 | 0.026 | 1.00 | 0.22 | 0 |
| global auto-gamma | 3.54 | 0.98 | 0.908 | 0.030 | 0.77 | 0.39 | 10 |
| global histogram eq. | 3.54 | 5.37 | 0.826 | 0.085 | 1.02 | 0.44 | 5 |
| **classical** | 3.45 | 0.77 | 0.933 | 0.017 | 0.97 | 0.33 | 177 |
| **deep learning** | 3.95 | 0.76 | 0.929 | 0.019 | 0.91 | 0.36 | 238 |
| **hybrid** | 3.44 | 0.86 | 0.936 | 0.017 | 0.91 | 0.35 | 592 |

**What the numbers say (honestly)**

* On the two **real** low-light test sets the **hybrid is best on SSIM, LPIPS and NIQE** and on edge preservation, with no over-exposure,
  and it has the lowest colour shift on LOL eval15. Its PSNR is within 0.4 dB of the deep branch on LOL eval15 and the best of all on LOL-v2 Real among the learned methods.
* **PSNR is not uniformly won by the hybrid.** The deep branch alone has the highest PSNR on LOL eval15 and on LOL-v2 Synthetic, and on LOL-v2 Real a plain global
  auto-gamma curve scores a higher PSNR than every method (gamma 18.9 dB vs hybrid 18.2 dB), although with clearly worse SSIM (0.66 vs 0.73), LPIPS (0.39 vs 0.28) and NIQE (3.9 vs 3.1).
  PSNR rewards matching the overall brightness of the reference; LOL-v2 references vary in exposure.
* The deep branch alone is the **worst on NIQE** (about 5.4 on LOL, vs ~3.0-3.2 for classical/hybrid): it brightens well but leaves visibly noisier, less natural images.
* On **LOL-v2 Synthetic** the classical branch scores *below* global gamma on PSNR/SSIM (its adaptive denoising and tone mapping are tuned for real sensor noise, which this synthetic set does not contain),
  and deep learning wins; the hybrid sits between them. We did not tune anything on this set.
* On **unseen** images (no reference) all three adaptive methods keep over-exposure no higher than the input while global histogram equalisation roughly triples it. The classical/hybrid methods have the best (lowest) NIQE of the adaptive methods on most sets;
  however the **untouched input has the lowest NIQE on 5 of the 6 unseen sets** (all but NPE). NIQE rewards natural-image statistics and is known to be a weak judge of under-exposed images, so we treat it as a sanity check against artefacts, not as proof that enhancement is worse; the over-exposure, detail-ratio and brightness columns are the more informative ones here.
* **Latency** (CPU, <= 600 px, includes the whole pipeline): classical ~0.15 s, deep ~0.2 s, hybrid ~0.4-0.5 s on average (mean over all sets, laptop CPU) - all well inside the 10-second interactive target.

### 5.2 Ablation study

Evaluated on the 30 first paired images pooled over LOL eval15 / LOL-v2 Real / LOL-v2 Synthetic (`results/final/ablation.csv`). Each row removes or replaces one component; the last columns give the change in PSNR against the full pipeline of the same family.

| Variant | PSNR | SSIM | LPIPS | NIQE | Noise suppr. | Edge pres. | dPSNR |
|---|---|---|---|---|---|---|---|
| hybrid (full) | 17.92 | 0.698 | 0.291 | 3.11 | 2.37 | 0.845 | - |
| hybrid - adaptive parameters (fixed) | 16.91 | 0.737 | 0.275 | 2.78 | 6.93 | 0.860 | -1.01 |
| hybrid - illumination prior | 18.01 | 0.707 | 0.282 | 3.14 | 2.40 | 0.851 | +0.09 |
| hybrid - pyramid blending (alpha blend) | 17.94 | 0.699 | 0.291 | 3.10 | 2.38 | 0.845 | +0.02 |
| hybrid - DL pre-denoise | 17.88 | 0.624 | 0.316 | 4.69 | 1.47 | 0.833 | -0.04 |
| hybrid - DL detail shrinkage | 17.92 | 0.690 | 0.293 | 3.37 | 2.14 | 0.844 | -0.00 |
| hybrid - output guard (highlights/exposure) | 17.92 | 0.698 | 0.291 | 3.11 | 2.37 | 0.845 | +0.00 |
| hybrid - classical denoise | 17.39 | 0.453 | 0.467 | 6.83 | 0.73 | 0.763 | -0.53 |
| hybrid - classical shrinkage | 17.89 | 0.642 | 0.330 | 4.13 | 1.59 | 0.831 | -0.03 |
| hybrid - tone stretch | 17.43 | 0.726 | 0.264 | 2.91 | 2.70 | 0.853 | -0.49 |
| hybrid - CLAHE | 17.40 | 0.735 | 0.238 | 2.53 | 3.87 | 0.864 | -0.52 |
| hybrid - colour correction | 18.13 | 0.703 | 0.291 | 3.12 | 2.39 | 0.846 | +0.21 |
| hybrid - deep branch (= classical only) | 17.05 | 0.647 | 0.334 | 3.27 | 1.96 | 0.825 | -0.87 |
| hybrid - classical branch (= DL + post only) | 18.36 | 0.738 | 0.246 | 2.93 | 3.81 | 0.860 | +0.44 |
| classical (full) | 17.05 | 0.647 | 0.334 | 3.27 | 1.96 | 0.825 | - |
| classical - adaptive parameters (fixed) | 14.74 | 0.677 | 0.303 | 2.67 | 6.64 | 0.843 | -2.31 |
| classical - denoise | 16.50 | 0.380 | 0.524 | 7.59 | 0.58 | 0.736 | -0.55 |
| classical - shrinkage | 17.03 | 0.575 | 0.380 | 4.57 | 1.26 | 0.808 | -0.03 |
| classical - tone stretch | 15.85 | 0.679 | 0.312 | 3.03 | 2.15 | 0.830 | -1.20 |
| classical - CLAHE | 15.98 | 0.695 | 0.256 | 2.26 | 4.06 | 0.857 | -1.07 |
| classical - colour correction | 17.32 | 0.656 | 0.338 | 3.28 | 1.98 | 0.825 | +0.27 |
| classical - guard | 17.05 | 0.647 | 0.334 | 3.27 | 1.96 | 0.825 | -0.00 |
| classical: bilateral instead of guided denoise | 17.05 | 0.608 | 0.341 | 3.60 | 1.96 | 0.805 | +0.00 |
| classical: MSRCP instead of illumination decomposition | 15.50 | 0.567 | 0.373 | 3.64 | 1.87 | 0.813 | -1.55 |
| deep learning (quality-aware) | 17.96 | 0.535 | 0.380 | 5.58 | 1.26 | 0.816 | - |
| deep learning - quality-awareness (raw Zero-DCE) | 17.96 | 0.535 | 0.380 | 5.58 | 1.26 | 0.816 | - |

**Reading the ablation**

* **Load-bearing components:** classical denoising (removing it: SSIM 0.65 -> 0.38 and NIQE 3.3 -> 7.6 for the classical pipeline; SSIM 0.70 -> 0.45 for the hybrid), the Wiener detail shrinkage (SSIM -0.07, NIQE +1.3), the illumination decomposition
  (MSRCP instead: -1.6 dB, colour shift +3.9), and the deep branch inside the hybrid (+0.9 dB PSNR, +0.05 SSIM over classical alone).
* **Tone stretch and CLAHE** raise PSNR (about +0.5 dB for the hybrid, +1.1 to +1.2 dB for classical) but *lower* the perceptual scores: without CLAHE, SSIM / LPIPS / NIQE all improve (hybrid SSIM 0.698 -> 0.735, NIQE 3.12 -> 2.53) because CLAHE adds local contrast and noise.
  This is a genuine PSNR-versus-perception trade-off; we keep CLAHE on because the brief requires it and PSNR improves, but it is the first parameter to lower if perceptual quality matters more.
* **Adaptive parameters:** for the classical pipeline adaptive selection is worth +2.3 dB PSNR over fixed parameters; for the hybrid +1.0 dB. The fixed-parameter variants have higher noise suppression and slightly better SSIM/LPIPS/NIQE
  because their fixed settings are more aggressive (darker, smoother output, mean luma 0.27-0.33 vs 0.39): they trade brightness fidelity for smoothness. Adaptivity's clear benefit is robustness across exposure levels (see 5.3), not a uniform metric win on these dark images.
* **Honest negative results.** The hybrid with the classical branch removed (deep learning + post-processing only) scores *higher* than the full hybrid on this pool (PSNR +0.4 dB, SSIM 0.738 vs 0.698, LPIPS 0.246 vs 0.291, NIQE 2.93 vs 3.12), with ~1.6x lower latency.
  The classical branch earns its place on very noisy / very dark regions and in robustness (fewer over-exposed pixels on unseen images), not in the average LOL score. **Colour correction does not help on LOL:** removing it *raises* PSNR by about 0.2 dB (hybrid) / 0.3 dB (classical) and *lowers* the colour shift (hybrid 11.9 -> 11.4 dE). The grey-world assumption is wrong for many LOL scenes, so we keep it only as a partial (strength <= 0.6) correction; re-tuning or disabling it for LOL-like data is the obvious next step.
  The illumination prior (+0.09 dB when removed), pyramid vs alpha blending (0.015 dB) and the output guard (0.000 dB) have no measurable effect on these dark images (the guard exists for bright inputs, see 5.3). The DL pre-denoise and DL detail shrinkage are PSNR-neutral but matter perceptually: removing the pre-denoise drops SSIM 0.698 -> 0.624 and raises NIQE 3.1 -> 4.7.

### 5.3 Image-quality-aware strategy (deep branch)

On LOL the quality-aware wrapper and the raw Zero-DCE rows are identical, because every LOL image is uniformly very dark (darkness ~1 -> full curve iterations, full strength, no exposure feedback): the wrapper is inert by design there.
Its purpose is to avoid over-brightening images that are *not* very dark. Measured on the better-lit unseen sets (`results/quality_aware_check.md`): output mean luma 0.40 vs 0.59 (NPE), 0.36 vs 0.52 (VV), 0.38 vs 0.57 (Fusion), and over-exposed pixels 0.020 vs 0.051 (VV), 0.016 vs 0.022 (NPE), 0.036 vs 0.060 (Fusion), 0.033 vs 0.049 (DICM) - i.e. 30-60 % fewer over-exposed pixels.

## 6. Failure-case analysis

`results/final/failure_analysis.md` lists the five lowest-PSNR hybrid outputs with side-by-side panels
(input | input x4 | classical | deep | hybrid | ground truth) and automatically measured diagnostics. Visual inspection of the worst cases shows these concrete failure modes:

1. **Reference / exposure mismatch on already-lit images (`r05772356t`, 8.2 dB, SSIM 0.55).** The input is a daylight scene that is already reasonably exposed (mean luma 0.36, non-uniformity 0.79);
   the adaptive strategy correctly changes it very little, but the synthetic "ground truth" is a washed-out, over-bright version, so PSNR is very low although the output looks natural.
   Cause: the metric measures agreement with a reference whose exposure is not what a viewer would choose; this is a limitation of full-reference evaluation, not of the enhancer. Mitigation: report NIQE/LPIPS alongside PSNR (we do).
2. **Dark, saturated colours are not restored (`r03193652t`, tulips, 10.5 dB, colour cast 0.25).** Deep-red petals come out dull brown where the reference is saturated red, and smooth green backgrounds show swirl-like
   artefacts from the guided-filter / shrinkage stages. Cause: when the input chroma has almost no signal, colour has to be hallucinated; the conservative colour correction (capped at strength 0.6, gain range 0.75-1.33)
   limits both over-saturation and recovery. A learned colour/chroma network would be the remedy.
3. **Extremely dark input, residual noise and tint (LOL `55`, mean luma 0.04).** The hybrid recovers the scene and is cleaner than either branch alone, but fine speckle remains in the shadows and whites keep a slight green cast
   (11.5 dB). Cause: the gain needed is so large that the per-pixel noise floor approaches the signal; the gain-aware denoiser removes most noise but trades some detail, and grey-world white balance only partially corrects a cast that varies across the image.
4. **Strongly non-uniform lighting (non-uniformity 0.8-1.0 in four of the five worst images).** Bright and dark areas coexist in one image; a single smoothing radius for the illumination map cannot match both the source boundary and the shadows,
   so shadows are under-lifted while bright regions are left alone. Per-region radii / multi-scale illumination estimation are the proposed fix (future work).
5. **PSNR alone would pick the wrong system (LOL-v2 Real).** Plain auto-gamma beats every method on PSNR there (18.9 vs 18.2 dB) but is clearly worse on SSIM (0.66 vs 0.73), LPIPS (0.39 vs 0.28) and NIQE (3.9 vs 3.1) and recovers less detail/edge structure (edge preservation 0.75 vs 0.86).

Not observed in our tests: halo artefacts around light sources or strong over-exposure (over-exposed pixel fraction is 0.000 on all three paired sets and not higher than the input on unseen sets). Pure colour-cast failures from coloured illuminants were not specifically tested.

## 7. Deployment and UX

Streamlit app (`app/streamlit_app.py`): upload (PNG/JPEG/TIFF/BMP/WebP, 25 MB cap, EXIF-aware, auto down-scaling), method selector
(hybrid / classical / deep / compare all), optional reference image (enables PSNR/SSIM/LPIPS), synthetic-sample generator for demos,
tabs for result + download, intermediate pipeline steps (illumination, reflectance, gain heat-map, denoise, shrinkage, CLAHE, colour,
fusion weights), measured statistics + chosen parameters + histograms, and metrics. Corrupt files, oversized files and processing errors
produce readable messages instead of stack traces; the model is cached across requests. Measured on a laptop CPU at <= 600 px (evaluation): classical ~0.15-0.25 s, deep ~0.25-0.3 s, hybrid ~0.4-0.8 s. Deployed on Streamlit Community Cloud (CPU only); free apps sleep when idle and need ~30 s to wake. Container image via `Dockerfile` (CPU PyTorch), no secrets.

## 8. Limitations, ethics, future work

See README. Future work: learned denoiser in front of the curves (e.g. a small U-Net trained on LOL), per-region radius for illumination
estimation, learned fusion weights, int8/fp16 export of the 79k-parameter network for mobile, RAW-domain processing.
