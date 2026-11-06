# Model weights

* `zero_dce.pt` - Zero-DCE checkpoint (79,416 parameters, ~330 KB) produced by `scripts/train.py`. **Commit it** after training so the
  deployed app can load it (override the location with the `LOWLIGHT_WEIGHTS` environment variable).
* `niqe_params.npz` - NIQE-style pristine model fitted by `scripts/fit_niqe.py` on well-exposed images (optional; without it
  the metric falls back to the official `pyiqa` parameters when internet is available).

Without `zero_dce.pt` the app only offers the classical pipeline and `scripts/evaluate.py` refuses to run (use `--allow-untrained` for smoke tests).
