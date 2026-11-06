# Image-quality-aware strategy: effect on the deep branch

Deep branch (Zero-DCE, trained checkpoint) with the quality-aware wrapper (QA) vs raw Zero-DCE (all 8 curve iterations,
no exposure feedback or highlight roll-off). First 20 images per set, resized to 400 px max side; no ground truth needed.
Script: `scripts/quality_aware_check.py` (the same `DLEnhancer(quality_aware=True/False)` used in `scripts/evaluate.py`).

| Set | n | input mean luma | QA output luma | raw output luma | QA over-exposed px | raw over-exposed px | mean abs diff QA vs raw |
|---|---|---|---|---|---|---|---|
| LOL eval15 (very dark) | 15 | 0.059 | 0.400 | 0.400 | 0.000 | 0.000 | 0.000 |
| DICM | 20 | 0.206 | 0.324 | 0.427 | 0.033 | 0.049 | 0.103 |
| NPE | 20 | 0.279 | 0.395 | 0.592 | 0.016 | 0.022 | 0.197 |
| VV | 20 | 0.267 | 0.355 | 0.521 | 0.020 | 0.051 | 0.166 |
| Fusion | 18 | 0.313 | 0.383 | 0.566 | 0.036 | 0.060 | 0.184 |
| MEF (very dark) | 20 | 0.064 | 0.274 | 0.287 | 0.000 | 0.001 | 0.013 |

Reading: on uniformly very dark inputs (LOL, MEF) the wrapper is effectively inert (darkness ~1 -> full iterations, full
strength, no feedback needed), which is why the "quality-aware vs raw" ablation rows on LOL are identical. On moderately lit inputs
(DICM, NPE, VV, Fusion) it lowers the output brightness towards the target exposure and cuts over-exposed pixels by about
30-60 %, i.e. it prevents the raw network from over-brightening images that were not very dark.
