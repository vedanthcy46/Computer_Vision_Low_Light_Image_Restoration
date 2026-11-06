# System architecture

```mermaid
flowchart TD
    IN[Input image<br/>any size / dtype / gray / RGBA] --> PRE[Validate · EXIF · to float RGB · max-side resize]
    PRE --> QA[Image-quality analysis<br/>exposure · noise σ · non-uniformity · colour cast]
    QA --> AP[Adaptive parameter selection<br/>γ · gain cap · denoise k · shrink · CLAHE · colour · strength · DL prior]
    PRE --> IL[Illumination estimation<br/>max-RGB + guided filter → L, R = I/L]

    subgraph CLASSICAL [Classical branch]
        direction TB
        C1[Illumination curve L' = L^γ<br/>gain G = L^(γ-1), capped] --> C2[Gain-aware guided denoise<br/>σ·G per-pixel threshold]
        C2 --> C3[Wiener detail shrinkage]
        C3 --> C4[Tone stretch] --> C5[CLAHE on Lab-L] --> C6[Grey-world + saturation] --> C7[Soft highlight roll-off]
    end

    subgraph DEEP [Deep branch]
        direction TB
        D1[Light guided pre-denoise] --> D2[Zero-DCE: 8 curve maps<br/>iterations ∝ darkness]
        D2 --> D3[Wiener shrinkage] --> D4[Colour correction]
    end

    IL --> C1
    AP --> C1
    AP --> D1
    PRE --> D1

    C7 --> FU
    D4 --> FU
    IL --> FP[Illumination prior<br/>dark regions → classical]
    AP --> FP
    FP --> FU[Adaptive Laplacian-pyramid fusion<br/>well-exposedness × contrast × saturation × prior]
    FU --> OG[Output guard<br/>highlight roll-off · exposure feedback · strength blend]
    OG --> OUT[Enhanced image + intermediates]

    OUT --> APP[Streamlit app<br/>result · pipeline steps · analysis · metrics]
    OUT --> EV[evaluate.py<br/>PSNR SSIM LPIPS NIQE latency noise edge<br/>comparison · ablation · failure analysis]

    TR[train.py<br/>zero-reference + supervised fine-tuning<br/>LOL · LOL-v2 · BrighteningTrain] -.weights/zero_dce.pt.-> D2
```

## Module map

| Stage | Module |
|---|---|
| Quality analysis / adaptive parameters | `lowlight/quality.py`, `lowlight/adaptive.py` |
| Illumination estimation, Retinex (decomposition and MSRCP) | `lowlight/retinex.py` |
| Guided/bilateral denoise, Wiener shrinkage, highlight roll-off, pyramid fusion | `lowlight/filters.py` |
| Classical pipeline | `lowlight/classical.py` |
| Zero-DCE model, losses, quality-aware inference | `lowlight/dl/` |
| Hybrid fusion | `lowlight/hybrid.py` |
| Method / ablation registry | `lowlight/variants.py` |
| Metrics (PSNR, SSIM, LPIPS, NIQE, noise, edges, ...) | `lowlight/metrics.py`, `lowlight/niqe.py` |
| Data discovery, synthetic degradation, torch datasets | `lowlight/data.py` |
| UI / deployment | `app/streamlit_app.py`, `Dockerfile` |
