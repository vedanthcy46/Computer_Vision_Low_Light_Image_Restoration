import numpy as np
import pytest

from lowlight import metrics as M
from lowlight.adaptive import fixed_params, select_params
from lowlight.classical import ClassicalConfig, ClassicalEnhancer
from lowlight.data import degrade_low_light, find_pairs
from lowlight.filters import denoise, guided_filter, pyramid_fuse, soft_highlight_compress, wiener_shrink
from lowlight.hybrid import HybridConfig, HybridEnhancer
from lowlight.quality import analyze, estimate_noise_sigma
from lowlight.retinex import decompose, estimate_illumination
from lowlight.utils import InvalidImageError, load_image, to_float
from lowlight.variants import ablation_variants, main_methods


# ------------------------------------------------------------------ input handling
@pytest.mark.parametrize("arr", [
    np.zeros((40, 50), np.uint8), np.zeros((40, 50, 1), np.uint8), np.zeros((40, 50, 4), np.uint8),
    np.zeros((40, 50, 3), np.uint16), np.zeros((40, 50, 3), np.float32), np.full((40, 50, 3), 255.0, np.float64),
])
def test_to_float_accepts_many_formats(arr):
    out = to_float(arr)
    assert out.shape == (40, 50, 3) and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1


@pytest.mark.parametrize("bad", [None, np.zeros((0, 3, 3)), np.zeros((4, 4, 2)), np.zeros((4,))])
def test_to_float_rejects_garbage(bad):
    with pytest.raises(InvalidImageError):
        to_float(bad)


def test_load_image_rejects_corrupt_bytes():
    with pytest.raises(InvalidImageError):
        load_image(b"this is not an image")


# --------------------------------------------------------------------------- analysis
def test_analysis_orders_darkness(pairs):
    lo, gt = pairs[0]
    assert analyze(lo).darkness > analyze(gt).darkness
    assert analyze(lo).mean_lum < analyze(gt).mean_lum


def test_noise_estimator_tracks_noise():
    rng = np.random.default_rng(0)
    base = np.full((128, 128, 3), 0.3, np.float32)
    s1 = estimate_noise_sigma(np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1).astype(np.float32))
    s2 = estimate_noise_sigma(np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1).astype(np.float32))
    # luma averages three independent channels -> noise std is ~0.75 x per-channel sigma
    assert 0.006 < s1 < 0.009 and 0.03 < s2 < 0.045 and s2 > 3 * s1


def test_adaptive_params_follow_image_quality(pairs):
    dark_p = select_params(analyze(pairs[0][0]))
    bright_p = select_params(analyze(pairs[0][1]))
    assert dark_p.strength > bright_p.strength            # well-exposed image: leave alone
    assert dark_p.gamma <= bright_p.gamma
    noisy = analyze(degrade_low_light(pairs[0][1], np.random.default_rng(0), 0.9))
    clean = type(noisy)(**{**noisy.as_dict(), "noise_level": 0.0})
    noisy.noise_level = 1.0
    assert select_params(noisy).gain_max < select_params(clean).gain_max   # noisy -> capped gain


# ------------------------------------------------------------------------------ filters
def test_guided_filter_preserves_step_edge():
    img = np.zeros((64, 64), np.float32)
    img[:, 32:] = 1.0
    out = guided_filter(img, img, 4, 1e-3)
    assert abs(out[:, 28] - 0).max() < 0.05 and abs(out[:, 36] - 1).max() < 0.05


def test_denoise_reduces_noise_and_accepts_sigma_map():
    rng = np.random.default_rng(1)
    clean = np.full((96, 96, 3), 0.4, np.float32)
    noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1).astype(np.float32)
    for sigma in (0.05, np.full((96, 96), 0.05, np.float32)):
        out = denoise(noisy, sigma)
        assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean() * 0.5


def test_wiener_shrink_keeps_strong_edges():
    img = np.zeros((64, 64, 3), np.float32)
    img[:, 32:] = 0.8
    out = wiener_shrink(img, np.full((64, 64), 0.01, np.float32))
    assert out[:, 40].mean() > 0.7 and out[:, 20].mean() < 0.1


def test_highlight_compression_is_monotone_and_bounded():
    x = np.linspace(0, 1.5, 200).astype(np.float32)
    y = soft_highlight_compress(x)
    assert (np.diff(y) >= -1e-6).all() and y.max() <= 1.0


def test_pyramid_fuse_of_identical_images_is_identity():
    rng = np.random.default_rng(0)
    im = rng.random((64, 64, 3)).astype(np.float32)
    w = np.ones((64, 64), np.float32)
    assert np.abs(pyramid_fuse([im, im], [w, w]) - im).max() < 1e-4


def test_illumination_bounds_and_reflectance(dark):
    lum, refl = decompose(dark)
    assert lum.min() >= 1e-3 and lum.max() <= 1.0
    assert (lum >= dark.max(axis=2) - 1e-6).all()          # L >= max channel -> R <= 1
    assert refl.max() <= 1.0 + 1e-6


# ----------------------------------------------------------------------------- pipelines
def test_classical_brightens_dark_input_in_range(dark):
    res = ClassicalEnhancer().enhance(dark)
    out = res.image
    assert out.shape == dark.shape and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
    assert out.mean() > dark.mean() * 1.5
    for k in ("illumination", "reflectance", "gain", "denoised", "shrunk", "clahe", "color", "final"):
        assert k in res.stages


def test_well_exposed_image_is_not_overexposed(pairs):
    gt = pairs[0][1]
    out = ClassicalEnhancer().enhance(gt).image
    assert abs(out.mean() - gt.mean()) < 0.08
    assert M.overexposure(out, gt) < 0.02


def test_all_ablation_variants_run(dl, dark):
    small = dark[:64, :64]
    for name, fn in ablation_variants(dl).items():
        out = np.asarray(getattr(fn(small), "image", fn(small)))
        assert out.shape == small.shape and np.isfinite(out).all(), name
    for name, fn in main_methods(dl).items():
        assert fn(small).shape == small.shape, name


@pytest.mark.parametrize("make", [
    lambda: np.zeros((96, 96, 3), np.float32), lambda: np.ones((96, 96, 3), np.float32),
    lambda: np.full((96, 96, 3), 0.02, np.float32), lambda: np.random.default_rng(0).random((17, 23, 3)).astype(np.float32),
    lambda: np.zeros((3, 3, 3), np.float32),
])
def test_degenerate_inputs_do_not_crash(dl, make):
    img = make()
    for enh in (ClassicalEnhancer(), HybridEnhancer(dl)):
        out = enh.enhance(img).image
        assert out.shape == img.shape and np.isfinite(out).all()


def test_hybrid_modes_and_weight_map(dl, dark):
    res = HybridEnhancer(dl).enhance(dark)
    assert "w_dl" in res.stages and 0 <= res.stages["w_dl"].min() and res.stages["w_dl"].max() <= 1
    only_c = HybridEnhancer(dl, HybridConfig(use_dl=False)).enhance(dark).image
    assert np.abs(only_c - ClassicalEnhancer().enhance(dark).image).mean() < 0.02   # same branch, only the output guard differs


def test_fixed_vs_adaptive_params_differ(dark):
    a = ClassicalEnhancer(adaptive=True).enhance(dark).params
    b = ClassicalEnhancer(adaptive=False).enhance(dark).params
    assert a.gamma != b.gamma or a.gain_max != b.gain_max
    assert b.gamma == fixed_params().gamma


# ----------------------------------------------------------------------------- metrics
def test_metrics_sanity(pairs):
    lo, gt = pairs[0]
    assert M.psnr(gt, gt) == 100.0 and M.ssim(gt, gt) == pytest.approx(1.0)
    assert M.psnr(gt, lo) < 20
    assert M.edge_preservation(gt, gt) == pytest.approx(1.0, abs=1e-5)
    assert M.colour_shift(gt, gt) == 0.0
    out = ClassicalEnhancer().enhance(lo).image
    assert M.noise_suppression(lo, out) > 1.0                   # classical pipeline suppresses noise
    r = M.evaluate_pair(lo, out, gt, 12.0)
    for k in M.METRIC_ORDER + M.EXTRA_ORDER:
        assert k in r


def test_find_pairs_lol_layouts(tmp_path):
    from PIL import Image
    for sub, pref in (("Low", "low"), ("Normal", "normal")):
        d = tmp_path / "LOL-v2" / "Test" / sub
        d.mkdir(parents=True)
        for i in (1, 2):
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / f"{pref}0000{i}.png")
    for sub in ("low", "high"):
        d = tmp_path / "LOL" / "eval15" / sub
        d.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "111.png")
    assert len(find_pairs(tmp_path / "LOL-v2")) == 2
    assert len(find_pairs(tmp_path / "LOL")) == 1
