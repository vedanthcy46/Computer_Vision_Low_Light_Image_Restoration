1RUA24CSE0525,1RUA24CSE0544

# eval_P1_VisionX - Robust Low-Light Image Restoration and Enhancement

Metric order: PSNR ↑, SSIM ↑, LPIPS ↓, NIQE ↓, Latency_ms ↓ (ms), NoiseSuppression ↑, EdgePreservation ↑.
Run date 2026-11-06; DL weights: trained checkpoint; LPIPS AlexNet; NIQE backend `auto`.

## data_LOL_eval15 (15 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | 7.773 | 0.191 | 0.560 | 2.564 | 0.0 | 0.975 | 0.834 |
| gamma | 16.193 | 0.637 | 0.404 | 3.767 | 13.6 | 2.333 | 0.756 |
| hist_eq | 13.956 | 0.411 | 0.590 | 5.843 | 3.5 | 0.762 | 0.750 |
| classical | 17.050 | 0.663 | 0.328 | 3.205 | 126.2 | 1.886 | 0.848 |
| deep_learning | 18.539 | 0.575 | 0.351 | 5.351 | 210.1 | 1.305 | 0.843 |
| hybrid | 18.158 | 0.716 | 0.284 | 3.006 | 436.4 | 2.393 | 0.870 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.000 | 13.425 | 0.254 | 0.060 |
| gamma | 0.000 | 13.600 | 0.928 | 0.410 |
| hist_eq | 0.035 | 13.443 | 2.100 | 0.469 |
| classical | 0.000 | 10.044 | 1.370 | 0.394 |
| deep_learning | 0.000 | 12.012 | 1.293 | 0.409 |
| hybrid | 0.000 | 9.945 | 1.222 | 0.402 |

## data_LOL-v2_Real_captured_Test (30 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | 7.926 | 0.182 | 0.557 | 2.605 | 0.0 | 0.977 | 0.757 |
| gamma | 18.862 | 0.660 | 0.392 | 3.930 | 7.5 | 2.187 | 0.754 |
| hist_eq | 15.977 | 0.462 | 0.520 | 5.743 | 2.9 | 0.806 | 0.766 |
| classical | 17.193 | 0.682 | 0.318 | 3.246 | 96.8 | 1.915 | 0.837 |
| deep_learning | 17.986 | 0.549 | 0.388 | 5.428 | 167.2 | 1.185 | 0.816 |
| hybrid | 18.158 | 0.729 | 0.275 | 3.096 | 416.6 | 2.408 | 0.857 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.000 | 7.277 | 0.324 | 0.063 |
| gamma | 0.000 | 9.576 | 0.998 | 0.411 |
| hist_eq | 0.025 | 7.265 | 2.221 | 0.482 |
| classical | 0.000 | 11.560 | 1.393 | 0.371 |
| deep_learning | 0.000 | 12.921 | 1.481 | 0.404 |
| hybrid | 0.000 | 11.610 | 1.280 | 0.388 |

## data_LOL-v2_Synthetic_Test (30 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | 10.347 | 0.407 | 0.407 | 3.377 | 0.0 | 0.958 | 0.742 |
| gamma | 16.134 | 0.785 | 0.217 | 3.825 | 6.8 | 1.170 | 0.914 |
| hist_eq | 15.304 | 0.723 | 0.308 | 4.200 | 3.2 | 0.764 | 0.902 |
| classical | 14.901 | 0.717 | 0.257 | 3.458 | 95.0 | 0.821 | 0.903 |
| deep_learning | 16.954 | 0.792 | 0.179 | 4.161 | 149.3 | 0.825 | 0.932 |
| hybrid | 16.309 | 0.769 | 0.215 | 3.667 | 333.6 | 0.948 | 0.920 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.001 | 14.173 | 0.752 | 0.187 |
| gamma | 0.001 | 12.738 | 0.843 | 0.391 |
| hist_eq | 0.042 | 14.290 | 1.489 | 0.468 |
| classical | 0.000 | 9.704 | 1.156 | 0.345 |
| deep_learning | 0.000 | 8.739 | 1.119 | 0.384 |
| hybrid | 0.000 | 8.769 | 1.110 | 0.369 |

## unseen_DICM (30 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | n/a | n/a | n/a | 2.930 | 0.0 | 0.943 | 1.000 |
| gamma | n/a | n/a | n/a | 3.691 | 12.6 | 0.921 | 0.862 |
| hist_eq | n/a | n/a | n/a | 3.440 | 6.1 | 2.601 | 0.852 |
| classical | n/a | n/a | n/a | 3.750 | 202.8 | 0.645 | 0.907 |
| deep_learning | n/a | n/a | n/a | 4.175 | 263.9 | 0.659 | 0.901 |
| hybrid | n/a | n/a | n/a | 3.712 | 737.2 | 0.714 | 0.909 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.028 | n/a | 1.000 | 0.184 |
| gamma | 0.031 | n/a | 0.755 | 0.375 |
| hist_eq | 0.085 | n/a | 0.984 | 0.392 |
| classical | 0.023 | n/a | 0.968 | 0.309 |
| deep_learning | 0.024 | n/a | 0.926 | 0.338 |
| hybrid | 0.023 | n/a | 0.915 | 0.326 |

## unseen_LIME (10 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | n/a | n/a | n/a | 3.512 | 0.0 | 0.953 | 1.000 |
| gamma | n/a | n/a | n/a | 4.220 | 9.2 | 1.186 | 0.912 |
| hist_eq | n/a | n/a | n/a | 4.065 | 4.6 | 0.797 | 0.873 |
| classical | n/a | n/a | n/a | 4.681 | 170.2 | 0.711 | 0.910 |
| deep_learning | n/a | n/a | n/a | 4.957 | 219.7 | 0.822 | 0.906 |
| hybrid | n/a | n/a | n/a | 4.455 | 539.7 | 0.888 | 0.915 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.005 | n/a | 1.000 | 0.143 |
| gamma | 0.012 | n/a | 0.665 | 0.385 |
| hist_eq | 0.132 | n/a | 0.988 | 0.476 |
| classical | 0.000 | n/a | 1.033 | 0.321 |
| deep_learning | 0.000 | n/a | 0.897 | 0.372 |
| hybrid | 0.000 | n/a | 0.910 | 0.353 |

## unseen_MEF (30 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | n/a | n/a | n/a | 2.401 | 0.0 | 0.956 | 1.000 |
| gamma | n/a | n/a | n/a | 4.053 | 7.0 | 0.962 | 0.784 |
| hist_eq | n/a | n/a | n/a | 4.723 | 3.8 | 27.277 | 0.763 |
| classical | n/a | n/a | n/a | 3.582 | 119.6 | 0.877 | 0.901 |
| deep_learning | n/a | n/a | n/a | 5.118 | 203.8 | 0.614 | 0.894 |
| hybrid | n/a | n/a | n/a | 3.656 | 430.3 | 0.906 | 0.910 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.018 | n/a | 1.000 | 0.115 |
| gamma | 0.019 | n/a | 0.782 | 0.355 |
| hist_eq | 0.047 | n/a | 1.038 | 0.385 |
| classical | 0.012 | n/a | 0.858 | 0.277 |
| deep_learning | 0.013 | n/a | 0.888 | 0.304 |
| hybrid | 0.013 | n/a | 0.839 | 0.292 |

## unseen_NPE (30 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | n/a | n/a | n/a | 3.778 | 0.0 | 0.953 | 1.000 |
| gamma | n/a | n/a | n/a | 3.986 | 9.6 | 1.174 | 0.963 |
| hist_eq | n/a | n/a | n/a | 3.848 | 5.0 | 0.610 | 0.905 |
| classical | n/a | n/a | n/a | 3.662 | 179.4 | 0.904 | 0.955 |
| deep_learning | n/a | n/a | n/a | 3.920 | 257.0 | 1.031 | 0.960 |
| hybrid | n/a | n/a | n/a | 3.699 | 578.0 | 1.023 | 0.960 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.025 | n/a | 1.000 | 0.300 |
| gamma | 0.027 | n/a | 0.790 | 0.430 |
| hist_eq | 0.074 | n/a | 1.318 | 0.477 |
| classical | 0.018 | n/a | 1.029 | 0.377 |
| deep_learning | 0.021 | n/a | 0.910 | 0.419 |
| hybrid | 0.018 | n/a | 0.934 | 0.404 |

## unseen_VV (24 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | n/a | n/a | n/a | 2.013 | 0.0 | 0.963 | 1.000 |
| gamma | n/a | n/a | n/a | 2.363 | 12.3 | 0.764 | 0.950 |
| hist_eq | n/a | n/a | n/a | 2.519 | 6.4 | 0.480 | 0.741 |
| classical | n/a | n/a | n/a | 2.193 | 229.9 | 0.753 | 0.951 |
| deep_learning | n/a | n/a | n/a | 2.578 | 265.8 | 0.642 | 0.939 |
| hybrid | n/a | n/a | n/a | 2.202 | 737.0 | 0.807 | 0.948 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.030 | n/a | 1.000 | 0.260 |
| gamma | 0.035 | n/a | 0.760 | 0.380 |
| hist_eq | 0.072 | n/a | 0.912 | 0.458 |
| classical | 0.017 | n/a | 0.948 | 0.320 |
| deep_learning | 0.019 | n/a | 0.895 | 0.362 |
| hybrid | 0.017 | n/a | 0.896 | 0.348 |

## unseen_Fusion (18 images)

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation |
|---|---|---|---|---|---|---|---|
| input | n/a | n/a | n/a | 2.814 | 0.0 | 0.938 | 1.000 |
| gamma | n/a | n/a | n/a | 2.956 | 8.9 | 0.879 | 0.976 |
| hist_eq | n/a | n/a | n/a | 2.673 | 4.8 | 0.469 | 0.824 |
| classical | n/a | n/a | n/a | 2.833 | 160.2 | 0.757 | 0.977 |
| deep_learning | n/a | n/a | n/a | 2.927 | 216.1 | 0.782 | 0.972 |
| hybrid | n/a | n/a | n/a | 2.937 | 532.7 | 0.833 | 0.974 |

Additional diagnostics:

| Method | Overexposure | ColourShift_dE | DetailRatio | MeanLuma |
|---|---|---|---|---|
| input | 0.050 | n/a | 1.000 | 0.317 |
| gamma | 0.055 | n/a | 0.861 | 0.408 |
| hist_eq | 0.098 | n/a | 0.885 | 0.471 |
| classical | 0.032 | n/a | 0.980 | 0.359 |
| deep_learning | 0.038 | n/a | 0.958 | 0.390 |
| hybrid | 0.032 | n/a | 0.960 | 0.380 |


## Ablation study

Every row removes (or swaps) exactly one component of the full system; the `dPSNR` columns give the change relative to the full pipeline of the same family (negative = the component helps). Evaluated on 30 images with ground truth.

| Method | PSNR | SSIM | LPIPS | NIQE | Latency_ms | NoiseSuppression | EdgePreservation | Overexposure | ColourShift_dE | dPSNR vs hybrid | dPSNR vs classical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid (full) | 17.921 | 0.698 | 0.291 | 3.115 | 494.4 | 2.368 | 0.845 | 0.000 | 11.894 | 0.000 | n/a |
| hybrid - adaptive parameters (fixed) | 16.910 | 0.737 | 0.275 | 2.783 | 510.4 | 6.928 | 0.860 | 0.000 | 9.330 | -1.011 | n/a |
| hybrid - illumination prior | 18.011 | 0.707 | 0.282 | 3.140 | 492.2 | 2.398 | 0.851 | 0.000 | 12.016 | 0.090 | n/a |
| hybrid - pyramid blending (alpha blend) | 17.936 | 0.699 | 0.291 | 3.104 | 481.7 | 2.385 | 0.845 | 0.000 | 11.892 | 0.015 | n/a |
| hybrid - DL pre-denoise | 17.877 | 0.624 | 0.316 | 4.691 | 465.6 | 1.473 | 0.833 | 0.000 | 12.236 | -0.044 | n/a |
| hybrid - DL detail shrinkage | 17.917 | 0.690 | 0.293 | 3.366 | 472.4 | 2.141 | 0.844 | 0.000 | 11.910 | -0.004 | n/a |
| hybrid - output guard (highlights/exposure) | 17.921 | 0.698 | 0.291 | 3.115 | 484.5 | 2.368 | 0.845 | 0.000 | 11.894 | 0.000 | n/a |
| hybrid - classical denoise | 17.395 | 0.453 | 0.467 | 6.833 | 468.8 | 0.734 | 0.763 | 0.000 | 13.822 | -0.526 | n/a |
| hybrid - classical shrinkage | 17.891 | 0.642 | 0.330 | 4.132 | 501.6 | 1.593 | 0.831 | 0.000 | 11.824 | -0.030 | n/a |
| hybrid - tone stretch | 17.434 | 0.726 | 0.264 | 2.913 | 518.7 | 2.705 | 0.853 | 0.000 | 10.405 | -0.487 | n/a |
| hybrid - CLAHE | 17.399 | 0.735 | 0.238 | 2.527 | 507.4 | 3.872 | 0.864 | 0.000 | 12.004 | -0.522 | n/a |
| hybrid - colour correction | 18.131 | 0.703 | 0.291 | 3.118 | 512.7 | 2.390 | 0.846 | 0.000 | 11.365 | 0.210 | n/a |
| hybrid - deep branch (= classical only) | 17.053 | 0.647 | 0.334 | 3.268 | 169.2 | 1.962 | 0.825 | 0.000 | 11.948 | -0.868 | n/a |
| hybrid - classical branch (= DL + post only) | 18.357 | 0.738 | 0.246 | 2.927 | 311.7 | 3.810 | 0.860 | 0.000 | 12.141 | 0.436 | n/a |
| classical (full) | 17.053 | 0.647 | 0.334 | 3.268 | 140.7 | 1.961 | 0.825 | 0.000 | 11.949 | n/a | 0.000 |
| classical - adaptive parameters (fixed) | 14.740 | 0.677 | 0.303 | 2.673 | 151.8 | 6.640 | 0.843 | 0.000 | 8.626 | n/a | -2.313 |
| classical - denoise | 16.503 | 0.380 | 0.524 | 7.588 | 114.4 | 0.578 | 0.736 | 0.000 | 15.369 | n/a | -0.549 |
| classical - shrinkage | 17.027 | 0.575 | 0.380 | 4.573 | 120.9 | 1.262 | 0.808 | 0.000 | 11.889 | n/a | -0.026 |
| classical - tone stretch | 15.854 | 0.679 | 0.312 | 3.028 | 137.4 | 2.152 | 0.830 | 0.000 | 9.470 | n/a | -1.199 |
| classical - CLAHE | 15.983 | 0.695 | 0.256 | 2.265 | 134.9 | 4.057 | 0.857 | 0.000 | 12.023 | n/a | -1.069 |
| classical - colour correction | 17.322 | 0.656 | 0.338 | 3.280 | 131.5 | 1.977 | 0.825 | 0.000 | 11.111 | n/a | 0.270 |
| classical - guard | 17.052 | 0.647 | 0.334 | 3.268 | 132.7 | 1.961 | 0.825 | 0.000 | 11.951 | n/a | -0.000 |
| classical: bilateral instead of guided denoise | 17.055 | 0.608 | 0.341 | 3.600 | 131.2 | 1.960 | 0.805 | 0.000 | 11.766 | n/a | 0.002 |
| classical: MSRCP instead of illumination decomposition | 15.499 | 0.567 | 0.373 | 3.636 | 297.3 | 1.869 | 0.813 | 0.000 | 15.868 | n/a | -1.554 |
| deep learning (quality-aware) | 17.958 | 0.535 | 0.380 | 5.581 | 227.6 | 1.262 | 0.816 | 0.000 | 13.541 | n/a | n/a |
| deep learning - quality-awareness (raw Zero-DCE) | 17.958 | 0.535 | 0.380 | 5.580 | 213.1 | 1.262 | 0.816 | 0.000 | 13.541 | n/a | n/a |

## Failure-case analysis

See `failure_analysis.md` / `docs/design.md`.
