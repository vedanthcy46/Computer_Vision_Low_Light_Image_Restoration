"""Streamlit application: adaptive low-light restoration (classical / deep learning / hybrid)."""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight import metrics as M  # noqa: E402
from lowlight.adaptive import fixed_params  # noqa: E402
from lowlight.classical import ClassicalConfig, ClassicalEnhancer  # noqa: E402
from lowlight.data import bundled_images, degrade_low_light  # noqa: E402
from lowlight.dl.infer import DLEnhancer, load_model  # noqa: E402
from lowlight.hybrid import HybridConfig, HybridEnhancer  # noqa: E402
from lowlight.quality import analyze  # noqa: E402
from lowlight.utils import InvalidImageError, load_image, to_uint8  # noqa: E402
from lowlight.viz import NICE_NAMES, histogram_figure, to_display  # noqa: E402

MAX_BYTES = 25 * 1024 * 1024
st.set_page_config(page_title="Low-Light Restoration", page_icon="🌙", layout="wide")


def _weights_stamp() -> tuple:
    """Changes whenever the checkpoint changes, so a redeploy with new weights is never served from a stale cache."""
    from lowlight.dl.infer import DEFAULT_WEIGHTS
    try:
        s = DEFAULT_WEIGHTS.stat()
        return (s.st_size, int(s.st_mtime))
    except OSError:
        return (0, 0)


@st.cache_resource(show_spinner="Loading models...")
def get_models(stamp: tuple):  # `stamp` is part of the cache key (no leading underscore)
    model, loaded = load_model()
    return DLEnhancer(model=model), loaded


def build(method: str, dl: DLEnhancer, adaptive: bool):
    if method == "Classical":
        return ClassicalEnhancer(ClassicalConfig(adaptive=adaptive))
    if method == "Deep learning (Zero-DCE)":
        return dl
    return HybridEnhancer(dl, HybridConfig(adaptive=adaptive))


def png_bytes(img: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(to_uint8(img)).save(b, format="PNG")
    return b.getvalue()


def metrics_row(inp, out, ms, gt=None):
    r = M.evaluate_pair(inp, out, gt, ms, "auto")
    return {k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}


# ------------------------------------------------------------------------------ UI
dl, weights_ok = get_models(_weights_stamp())
st.title("🌙 Adaptive Low-Light Image Restoration")
st.caption("Illumination-adaptive enhancement: classical Retinex/CLAHE · Zero-DCE deep curves · hybrid fusion")

with st.sidebar:
    st.header("Settings")
    options = ["Hybrid (recommended)", "Classical", "Deep learning (Zero-DCE)", "Compare all three"]
    if not weights_ok:
        st.warning("No trained Zero-DCE weights found (`weights/zero_dce.pt`). Deep-learning and hybrid modes "
                   "would use an untrained network, so only the classical pipeline is enabled. See README -> Training.")
        options = ["Classical"]
    method = st.radio("Method", options)
    max_side = st.slider("Max image side (px)", 400, 1600, 800, 100, help="Larger inputs are down-scaled to keep latency low.")
    adaptive = st.toggle("Adaptive parameters", True, help="Off = fixed default parameters (for comparison).")
    show_steps = st.toggle("Show intermediate steps", True)
    st.divider()
    st.subheader("No low-light image at hand?")
    sev = st.slider("Synthetic darkness", 0.0, 1.0, 0.8, 0.05)
    if st.button("Generate a synthetic sample"):
        rng = np.random.default_rng(int(time.time()))
        base = bundled_images(512)
        gt = base[int(rng.integers(len(base)))]
        st.session_state["sample"] = (degrade_low_light(gt, rng, severity=sev), gt)

up = st.file_uploader("Upload a low-light image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
ref_up = st.file_uploader("Optional: well-exposed reference (enables PSNR / SSIM / LPIPS)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])

img, gt = None, None
try:
    if up is not None:
        if up.size > MAX_BYTES:
            raise InvalidImageError("file is larger than 25 MB")
        img = load_image(up.getvalue(), max_side=max_side)
        if ref_up is not None:
            gt = load_image(ref_up.getvalue())
            if gt.shape != img.shape:
                import cv2
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    elif "sample" in st.session_state:
        img, gt = st.session_state["sample"]
        st.info("Showing the synthetic sample (its original is used as the reference).")
except InvalidImageError as e:
    st.error(f"Could not read that image: {e}. Please upload a PNG/JPEG/TIFF/BMP/WebP file under 25 MB.")
    st.stop()
except Exception as e:  # never show a stack trace to the user
    st.error(f"Unexpected problem while loading the image ({type(e).__name__}). Try another file.")
    st.stop()

if img is None:
    st.info("Upload an image (or generate a synthetic sample in the sidebar) to begin.")
    st.stop()

methods = ["Classical", "Deep learning (Zero-DCE)", "Hybrid (recommended)"] if method.startswith("Compare") else [method]
results = {}
try:
    with st.spinner("Enhancing..."):
        for m in methods:
            enh = build(m, dl, adaptive)
            t = time.perf_counter()
            res = enh.enhance(img)
            results[m] = (res, (time.perf_counter() - t) * 1000.0)
except Exception as e:
    st.error(f"Enhancement failed ({type(e).__name__}: {e}). Try a smaller image or a different method.")
    st.stop()

main = methods[-1] if not method.startswith("Compare") else "Hybrid (recommended)"
res, ms = results[main]
stats, params = res.stats, res.params

tab_res, tab_steps, tab_an, tab_met = st.tabs(["Result", "Pipeline steps", "Analysis", "Metrics"])

with tab_res:
    if method.startswith("Compare"):
        cols = st.columns(len(results) + 1)
        cols[0].image(img, caption="Input", width="stretch")
        for c, (m, (r, t)) in zip(cols[1:], results.items()):
            c.image(r.image, caption=f"{m} - {t:.0f} ms", width="stretch")
    else:
        c1, c2 = st.columns(2)
        c1.image(img, caption="Input", width="stretch")
        c2.image(res.image, caption=f"{main} - {ms:.0f} ms", width="stretch")
    if gt is not None:
        st.image(gt, caption="Reference", width=320)
    st.download_button("Download result (PNG)", png_bytes(res.image), file_name="enhanced.png", mime="image/png")
    if stats.darkness < 0.15:
        st.success("The input is already well exposed; the adaptive strategy applied only a very light touch.")

with tab_steps:
    if not show_steps:
        st.info("Enable 'Show intermediate steps' in the sidebar.")
    else:
        keys = [k for k in res.stages if not k.startswith("classical/")]
        keys += [k for k in res.stages if k.startswith("classical/")]
        cols = st.columns(3)
        for i, k in enumerate(keys):
            label = NICE_NAMES.get(k.split("/")[-1], k)
            if k.startswith("classical/"):
                label = "Classical · " + label
            cols[i % 3].image(to_display(k.split("/")[-1], res.stages[k]), caption=label, width="stretch")

with tab_an:
    c1, c2 = st.columns(2)
    c1.subheader("Measured image quality")
    c1.dataframe({"statistic": list(stats.as_dict()), "value": [round(v, 4) for v in stats.as_dict().values()]}, hide_index=True)
    c2.subheader("Adaptive parameters chosen" if adaptive else "Fixed parameters")
    p = params.as_dict()
    c2.dataframe({"parameter": list(p), "value": [round(v, 4) for v in p.values()]}, hide_index=True)
    st.pyplot(histogram_figure({"input": img, "output": res.image}))

with tab_met:
    rows = {m: metrics_row(img, r.image, t, gt) for m, (r, t) in results.items()}
    st.dataframe(rows, width="stretch")
    st.caption("NoiseSuppression: noise of the naively-brightened input / noise of the output (higher is cleaner). "
               "EdgePreservation: correlation of edge maps (1 = identical). NIQE: lower is more natural"
               + ("; LPIPS/PSNR/SSIM need a reference." if gt is None else "."))
