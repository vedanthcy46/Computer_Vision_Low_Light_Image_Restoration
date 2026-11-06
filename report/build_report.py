"""Build the Hackathon Technical Solution Report (PDF) for team VisionX, problem P1.

    python report/build_report.py            # -> report/VisionX_P1_Solution_Report.pdf

All tables are read from results/final/summary.csv and results/final/ablation.csv (produced by scripts/evaluate.py);
figures come from scripts/make_report_figures.py and results/final/visuals.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle, Preformatted, CondPageBreak)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figs"
OUT = ROOT / "report" / "VisionX_P1_Solution_Report.pdf"

REPO = "https://github.com/vedanthcy46/Computer_Vision_Low_Light_Image_Restoration"
COMMIT = "aa2723f9c118e8476197f5e2f7376c7079a38619"
LIVE = "https://visionxlowlightrestoration.streamlit.app/"
TEAM = "VisionX"
MEMBERS = "Vedanth C Y (1RUA24CSE0525), Yashas B R (1RUA24CSE0544)"

S = pd.read_csv(ROOT / "results/final/summary.csv")
A = pd.read_csv(ROOT / "results/final/ablation.csv").set_index("method")

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
INK, ACC, GREY = colors.HexColor("#1b2430"), colors.HexColor("#1f4e8c"), colors.HexColor("#5b6675")
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.9, leading=11.8, textColor=INK, spaceAfter=3.5)
small = ParagraphStyle("small", parent=body, fontSize=7.9, leading=10.2, textColor=GREY, spaceAfter=2)
cap = ParagraphStyle("cap", parent=small, alignment=TA_CENTER, spaceBefore=1, spaceAfter=7)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=ACC, spaceBefore=10, spaceAfter=4, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10.4, leading=13, textColor=INK, spaceBefore=6, spaceAfter=2, keepWithNext=1)
title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=INK, spaceAfter=6)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
code = ParagraphStyle("code", parent=body, fontName="Courier", fontSize=7.3, leading=8.9, backColor=colors.HexColor("#f2f4f7"),
                      borderPadding=(4, 4, 4, 4), spaceBefore=3, spaceAfter=6)


def P(t, st=body):
    return Paragraph(t, st)


def bullets(items, st=body):
    return ListFlowable([ListItem(P(i, st), leftIndent=10, bulletColor=ACC) for i in items], bulletType="bullet", start="-", leftIndent=11,
                        bulletFontSize=8)


def tbl(data, widths, header=1, bold_cells=(), zebra=True, align_center_from=1, font=8.2):
    t = Table(data, colWidths=widths, repeatRows=header)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, header - 1), "Helvetica-Bold", font),
          ("BACKGROUND", (0, 0), (-1, header - 1), colors.HexColor("#e6edf6")), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("LINEBELOW", (0, 0), (-1, header - 1), 0.8, ACC), ("LINEBELOW", (0, -1), (-1, -1), 0.5, GREY),
          ("ALIGN", (align_center_from, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    if zebra:
        for r in range(header, len(data)):
            if (r - header) % 2 == 1:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fc")))
    for (c, r) in bold_cells:
        st.append(("FONT", (c, r), (c, r), "Helvetica-Bold", font))
    t.setStyle(TableStyle(st))
    return t


def fig(name, width_cm, caption):
    from PIL import Image as PI
    w, h = PI.open(FIG / name).size
    img = Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, cap)])


# ------------------------------------------------------------------ data helpers
def row(ds, method):
    return S[(S.dataset == ds) & (S.method == method)].iloc[0]


DS = [("data_LOL_eval15", "LOL eval15 (15)"), ("data_LOL-v2_Real_captured_Test", "LOL-v2 Real (30)"),
      ("data_LOL-v2_Synthetic_Test", "LOL-v2 Synth. (30)")]
M3 = [("classical", "Classical baseline"), ("deep_learning", "Deep learning only (Zero-DCE)"), ("hybrid", "Proposed hybrid system")]

bench = [["Dataset", "Approach / model variant", "PSNR dB up", "SSIM up", "LPIPS down", "NIQE down", "Latency ms"]]
bold = []
for di, (ds, dname) in enumerate(DS):
    best = {k: None for k in ("PSNR", "SSIM", "LPIPS", "NIQE")}
    sub = {m: row(ds, m) for m, _ in M3}
    best["PSNR"] = max(sub, key=lambda m: round(sub[m].PSNR, 2)); best["SSIM"] = max(sub, key=lambda m: round(sub[m].SSIM, 3))
    best["LPIPS"] = min(sub, key=lambda m: round(sub[m].LPIPS, 3)); best["NIQE"] = min(sub, key=lambda m: round(sub[m].NIQE, 2))
    for mi, (m, mname) in enumerate(M3):
        r = sub[m]
        bench.append([dname if mi == 0 else "", mname, f"{r.PSNR:.2f}", f"{r.SSIM:.3f}", f"{r.LPIPS:.3f}", f"{r.NIQE:.2f}", f"{r.Latency_ms:.0f}"])
        ri = len(bench) - 1
        for ci, k in ((2, "PSNR"), (3, "SSIM"), (4, "LPIPS"), (5, "NIQE")):
            if best[k] == m:
                bold.append((ci, ri))

UNS = ["unseen_DICM", "unseen_LIME", "unseen_MEF", "unseen_NPE", "unseen_VV", "unseen_Fusion"]
un = S[S.dataset.isin(UNS)].groupby("method")[["NIQE", "Overexposure", "EdgePreservation", "Latency_ms"]].mean()
unseen_tbl = [["Method (6 unseen sets, no ground truth)", "NIQE down", "Over-exposed px down", "Edge preservation up", "Latency (ms)"]]
for m, n in [("input", "Raw input"), ("hist_eq", "Global histogram equalisation")] + M3:
    unseen_tbl.append([n, f"{un.loc[m, 'NIQE']:.2f}", f"{un.loc[m, 'Overexposure'] * 100:.1f}%", f"{un.loc[m, 'EdgePreservation']:.3f}", f"{un.loc[m, 'Latency_ms']:.0f}"])

ABL = [("hybrid (full)", "Full hybrid system"), ("hybrid - adaptive parameters (fixed)", "Hybrid, fixed (non-adaptive) parameters"),
       ("hybrid - illumination prior", "Hybrid without illumination prior in fusion"), ("hybrid - pyramid blending (alpha blend)", "Hybrid, alpha blend instead of pyramid"),
       ("hybrid - DL pre-denoise", "Hybrid without DL-branch pre-denoise"), ("hybrid - classical denoise", "Hybrid without classical gain-aware denoise"),
       ("hybrid - CLAHE", "Hybrid without CLAHE"), ("hybrid - tone stretch", "Hybrid without tone stretch"),
       ("hybrid - colour correction", "Hybrid without colour correction"), ("hybrid - deep branch (= classical only)", "Hybrid without deep branch (= classical)"),
       ("hybrid - classical branch (= DL + post only)", "Hybrid without classical branch (= DL + post)"),
       ("classical - adaptive parameters (fixed)", "Classical, fixed parameters"), ("classical - denoise", "Classical without denoise"),
       ("classical - shrinkage", "Classical without Wiener detail shrinkage"), ("classical: MSRCP instead of illumination decomposition", "Classical, MSRCP instead of our decomposition")]
full_h, full_c = A.loc["hybrid (full)", "PSNR"], A.loc["classical (full)", "PSNR"]
abl = [["Variant (30 held-out paired images)", "PSNR (dB)", "SSIM", "LPIPS", "NIQE", "dPSNR vs full"]]
for k, n in ABL:
    r = A.loc[k]
    base = full_h if k.startswith("hybrid") else full_c
    abl.append([n, f"{r.PSNR:.2f}", f"{r.SSIM:.3f}", f"{r.LPIPS:.3f}", f"{r.NIQE:.2f}", "-" if k.endswith("(full)") else f"{r.PSNR - base:+.2f}"])
abl.insert(12, ["Classical pipeline (full)", f"{A.loc['classical (full)', 'PSNR']:.2f}", f"{A.loc['classical (full)', 'SSIM']:.3f}",
                f"{A.loc['classical (full)', 'LPIPS']:.3f}", f"{A.loc['classical (full)', 'NIQE']:.2f}", "-"])

h_e, c_e, d_e = row("data_LOL_eval15", "hybrid"), row("data_LOL_eval15", "classical"), row("data_LOL_eval15", "deep_learning")
h_r, c_r, d_r = row("data_LOL-v2_Real_captured_Test", "hybrid"), row("data_LOL-v2_Real_captured_Test", "classical"), row("data_LOL-v2_Real_captured_Test", "deep_learning")
gam_r = row("data_LOL-v2_Real_captured_Test", "gamma")


# ------------------------------------------------------------------ architecture diagram
def arch_fig(path):
    fig_, ax = plt.subplots(figsize=(12.5, 4.0))
    ax.set_xlim(0, 125); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, t, fc="#e6edf6", ec="#1f4e8c", fs=6.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color="#1b2430", linespacing=1.35)

    def arr(x1, y1, x2, y2, style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color="#5b6675", lw=1))

    box(1, 15, 14, 10, "Input image\n(any size, EXIF,\nuint8/16, RGBA)", fc="#f2f4f7", ec="#5b6675")
    box(19, 29, 21, 9, "Image-quality analysis\nexposure / noise sigma /\nnon-uniformity / colour cast", fc="#fff4dc", ec="#b8860b")
    box(44, 29, 21, 9, "Adaptive parameters\ngamma, gain cap, denoise k,\nCLAHE clip, strength", fc="#fff4dc", ec="#b8860b")
    box(19, 15, 21, 10, "Illumination estimate\nL = guided max-RGB\nR = I / L (Retinex)")
    box(44, 15, 34, 10, "Classical branch\ngain-aware guided denoise -> Wiener shrinkage\n-> tone stretch -> CLAHE -> colour -> roll-off")
    box(44, 1, 34, 10, "Deep branch (Zero-DCE, 79k params)\nlight pre-denoise -> 8 curve maps (iters ~ darkness)\n-> Wiener shrinkage -> colour", fc="#e4f3e8", ec="#2d7d46")
    box(83, 6, 23, 19, "Hybrid fusion\nweights = contrast x\nsaturation x well-exposed\nx illumination prior\n(from L; dark regions\nfavour classical)\nLaplacian-pyramid blend", fc="#f9e5e5", ec="#a63d3d")
    box(109, 8, 15, 9, "Output guard\nhighlight roll-off,\nexposure feedback", fc="#f2f4f7", ec="#5b6675")
    box(109, 22, 15, 9, "Enhanced image\n+ intermediates\n(app / CLI)", fc="#f2f4f7", ec="#5b6675")
    arr(15, 20, 19, 20); arr(29.5, 29, 29.5, 25); arr(40, 33.5, 44, 33.5); arr(54.5, 29, 58, 25); arr(40, 20, 44, 20)
    ax.plot([8, 8, 30], [15, 6, 6], color="#5b6675", lw=1); arr(30, 6, 44, 6)
    arr(78, 20, 83, 17); arr(78, 6, 83, 10); arr(106, 12.5, 109, 12.5); arr(116.5, 17, 116.5, 22)
    fig_.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_)


arch_fig(FIG / "fig0_architecture.png")
from PIL import Image as _PI  # noqa: E402
_PI.open(ROOT / "results/final/visuals/fail_r03193652t.png").convert("RGB").save(FIG / "fig4_failure.jpg", quality=88)

# ------------------------------------------------------------------ content
st = []
st.append(P("Hackathon Technical Solution Report", title))
hdr = [[P("<b>Problem ID</b>", cell), P("P1", cell)],
       [P("<b>Problem Title</b>", cell), P("Robust Low-Light Image Restoration and Enhancement", cell)],
       [P("<b>Team Name</b>", cell), P(TEAM, cell)],
       [P("<b>Team Members</b>", cell), P(MEMBERS, cell)],
       [P("<b>GitHub Repository</b>", cell), P(f'<link href="{REPO}" color="#1f4e8c">{REPO}</link><br/>Evaluated commit: <font name="Courier" size="7.6">{COMMIT}</font> '
                                                 "(trained weights + evaluation results). The deployed app runs the latest <font name=\"Courier\">master</font>.", cell)],
       [P("<b>Live Demo / Deployment</b>", cell), P(f'<link href="{LIVE}" color="#1f4e8c">{LIVE}</link> (Streamlit Community Cloud, CPU only; free apps sleep when idle, '
                                                    "so the first visit may take about 30 s)", cell)]]
t = Table(hdr, colWidths=[3.6 * cm, 13.8 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d3df")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e6edf6")),
                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
st += [t, Spacer(1, 4)]

# 1 -------------------------------------------------------------------
st.append(P("1. Problem Understanding &amp; Solution Strategy", h1))
st.append(bullets([
    "<b>Core technical challenge.</b> Restore images that are severely under-exposed (mean luminance about 0.06 on the LOL and LOL-v2 Real test inputs, as low as 0.01), noisy, colour-cast and "
    "unevenly lit, <i>without</i> over-exposure, halos or unnatural colour. Brightening needs per-pixel gains of up to about 16x, so sensor noise of std "
    "sigma becomes sigma x gain: the classical failure modes are noise amplification in shadows, colour shift from per-channel gain, loss of texture when "
    "denoising is too strong, and blown-out highlights when a global curve is applied to a scene with both dark and bright regions. Constraints: interactive "
    "latency (under about 10 s per image on a free CPU host), a small trainable model, and honest evaluation on held-out and unseen images.",
    "<b>Proposed methodology.</b> We measure each image (exposure, noise level, non-uniformity, colour cast) and derive <i>all</i> enhancement parameters from "
    "those statistics instead of using fixed constants. A noise-aware Retinex/CLAHE pipeline and a Zero-DCE network each enhance the image, and a "
    "Laplacian-pyramid fusion, guided by the classical illumination map, combines them region by region: dark, noisy regions lean on the explicitly "
    "noise-controlled classical branch, better-lit regions use the learned curves. Static baselines (one global gamma or histogram equalisation) over-brighten "
    "bright areas and amplify noise because they cannot adapt to the image or to the region."]))
st.append(fig("fig0_architecture.png", 16.0, "Figure 1. System architecture (classical branch, deep branch, illumination-guided fusion, output guard). "
                                              "Modules: lowlight/quality.py, adaptive.py, retinex.py, filters.py, classical.py, dl/, hybrid.py."))

# 2 -------------------------------------------------------------------
st.append(P("2. Technical Architecture &amp; Implementation", h1))
st.append(P("2.1 Classical algorithmic baseline", h2))
st.append(bullets([
    "<b>Algorithms utilised</b> (<font name=\"Courier\">retinex.py, filters.py, classical.py</font>): (i) Retinex illumination decomposition I = R x L with "
    "L = max(guided-filter(max-RGB), max-RGB), so that R = I/L is at most 1; (ii) a gain-capped illumination curve L' = L^gamma; (iii) <b>gain-aware guided "
    "(or bilateral) edge-preserving denoising</b> whose per-pixel regularisation follows the amplified noise level (k x sigma x gain)^2, with stronger "
    "chroma smoothing; (iv) Wiener-style detail shrinkage that keeps strong edges and attenuates noise-level detail; (v) percentile tone stretch, "
    "<b>CLAHE</b> on Lab lightness, partial grey-world colour correction with a saturation gain, and a soft highlight roll-off.",
    "<b>Adaptive selection logic</b> (<font name=\"Courier\">quality.py -> adaptive.py</font>): the image is summarised by mean/percentile luminance, "
    "dark-pixel fraction, a robust MAD-based noise sigma, non-uniformity (dynamic range of blurred log-luminance), colour cast and a 0-1 darkness score. "
    "Parameters are smooth functions of these: illumination exponent gamma = log(target)/log(mean max-RGB), gain cap = 5 + 11(1 - noise), denoise "
    "threshold k = 1.2 + 2 x noise, CLAHE clip = 1.2 + 2.8 x darkness x (1 - 0.6 x noise), overall strength about max(1.25 x darkness, dark fraction), so "
    "well-exposed inputs are left almost untouched."]))
st.append(P("2.2 Deep learning pipeline", h2))
st.append(bullets([
    "<b>Model architecture.</b> Zero-DCE re-implemented in PyTorch (<font name=\"Courier\">lowlight/dl/zero_dce.py</font>): a 7-layer CNN (32 channels, 79,416 "
    "parameters, about 320 KB) predicts 8 per-pixel curve maps applied iteratively as LE(x) = x + a x (1 - x). <i>Why:</i> it is light enough for CPU "
    "deployment, it can be trained without ground truth, and its curves are monotone and bounded in [0, 1], which limits over-exposure and colour artefacts "
    "compared with a generator network. <b>Image-quality-aware inference</b> (<font name=\"Courier\">dl/infer.py</font>): curve iterations scale with "
    "darkness, noisy inputs get a light pre-denoise, highlights are rolled off, and an exposure-feedback step pulls back results that are too bright.",
    "<b>Loss functions and training.</b> Stage 1, zero-reference: spatial-consistency (weight 1), exposure to 0.6 (10), colour-constancy (5) and "
    "curve-smoothness (200) losses, 15 epochs, Adam, learning rate 1e-4, batch 8, 256-px patches, on the low-light images of LOL (485), LOL-v2 Real "
    "(689) and Synthetic (900) train splits plus the 1,000-image BrighteningTrain set. Stage 2, supervised fine-tuning: L1 (1.0) + SSIM (0.5) + smoothness (20) "
    "on the paired train splits of LOL, LOL-v2 Real and LOL-v2 Synthetic (7,468 patches, 10 epochs, lr 5e-5). Validation loss fell from 0.606 to 0.405 "
    "(stage 1) and 0.330 to 0.324 (stage 2). Test splits were never used for training or model selection."]))
st.append(P("2.3 Hybrid integration &amp; adaptive mechanism", h2))
st.append(bullets([
    "<b>Fusion strategy</b> (<font name=\"Courier\">hybrid.py</font>): both branches enhance the same input. Each candidate gets a per-pixel weight "
    "contrast^0.6 x saturation^0.4 x well-exposedness (contrast is measured on smoothed luminance so noise is not rewarded), multiplied by an "
    "<b>illumination prior</b> derived from the classical illumination map: p_deep = bias x (0.4 + 0.6 x smoothstep(L, 0.03, 0.2)). Weights are "
    "guided-filter smoothed and candidates are blended in a Laplacian pyramid, which avoids halos and seams at region boundaries. The deep branch "
    "itself is fed the classical image statistics (pre-denoise threshold, iteration count) and shares the classical shrinkage and colour stages.",
    "<b>Decision logic.</b> Very dark regions (small L) down-weight the deep branch, because amplifying them with learned curves also amplifies noise, and "
    "rely on the gain-aware denoiser; well-lit regions up-weight the learned curves, which keep local contrast without the heavy smoothing. The deep-branch "
    "bias itself falls with the measured noise (0.8 - 0.5 x noise). An output guard rolls off highlights and blends back towards the input if the result "
    "would be brighter than the target exposure."]))

# 3 -------------------------------------------------------------------
st.append(P("3. Experimental Benchmarks &amp; Evaluation", h1))
st.append(P("3.1 Quantitative benchmarks", h2))
st.append(P("Held-out test splits only (never trained on); at most 30 images per set, resized to at most 600 px. LPIPS: AlexNet. NIQE: NIQE-style model fitted "
            "locally on 300 well-exposed images (so values are comparable within this report, not with official NIQE). Latency: full pipeline, single image, laptop CPU. "
            "Best value per dataset and metric in bold. Reproduce with <font name=\"Courier\">scripts/evaluate.py</font> (command in Section 5).", small))
st.append(tbl(bench, [3.1 * cm, 5.0 * cm, 2.1 * cm, 1.7 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm], bold_cells=bold, align_center_from=2))
st.append(Spacer(1, 4))
st.append(P(
    f"<b>Analysis.</b> On the real low-light sets the hybrid gives the best structural and perceptual scores (LOL eval15: SSIM {h_e.SSIM:.3f} vs {c_e.SSIM:.3f} classical and "
    f"{d_e.SSIM:.3f} deep; LPIPS {h_e.LPIPS:.3f} vs {c_e.LPIPS:.3f} and {d_e.LPIPS:.3f}; NIQE {h_e.NIQE:.2f} vs {d_e.NIQE:.2f} for deep learning alone) and the highest PSNR on LOL-v2 Real "
    f"({h_r.PSNR:.2f} dB vs {c_r.PSNR:.2f} classical and {d_r.PSNR:.2f} deep). The deep branch alone has the best PSNR on LOL eval15 and on LOL-v2 Synthetic but the worst NIQE "
    f"(about {d_e.NIQE:.1f}): it brightens well but leaves noisier, less natural images. A plain global gamma curve even beats every method on LOL-v2 Real PSNR "
    f"({gam_r.PSNR:.1f} dB) while being clearly worse on SSIM ({gam_r.SSIM:.2f} vs {h_r.SSIM:.2f}), LPIPS ({gam_r.LPIPS:.2f} vs {h_r.LPIPS:.2f}) and NIQE, so PSNR alone would pick the wrong system. "
    f"The hybrid costs about {h_e.Latency_ms / c_e.Latency_ms:.1f}x the classical latency ({h_e.Latency_ms:.0f} vs {c_e.Latency_ms:.0f} ms) but stays well under one second per image."))
st.append(P("<b>Unseen images (DICM, LIME, MEF, NPE, VV, Fusion; no ground truth, mean of the six per-set means).</b>", body))
st.append(tbl(unseen_tbl, [6.4 * cm, 2.4 * cm, 3.4 * cm, 3.2 * cm, 2.0 * cm]))
st.append(P("All adaptive methods keep over-exposed pixels below the raw input's rate (histogram equalisation roughly triples it). The untouched input has the lowest NIQE on "
            "5 of the 6 unseen sets: NIQE is a weak judge of under-exposed images, so we use it as an artefact check, not as proof of quality.", small))

st.append(P("3.2 Visual outputs &amp; intermediate representations", h2))
st.append(fig("fig1_extreme_lowlight.jpg", 16.6,
              "Figure 2. Extreme low light (held-out LOL / LOL-v2 test images; luma = mean luminance of the input): raw input vs classical, deep learning, hybrid and ground truth, with per-image PSNR/SSIM."))
st.append(fig("fig2_nonuniform_backlit.jpg", 12.6,
              "Figure 3. Non-uniform / backlit scenes (unseen images; rows: Fusion/10, VV/P1010860, LIME/9): bright regions (sky, lamp shade) are not blown out; the fraction of over-exposed pixels is printed under each panel."))
st.append(fig("fig3_intermediate.jpg", 16.2,
              "Figure 4. Intermediate states of the hybrid on LOL 146: illumination map, gain map (high gain where the scene is darkest, hence where noise is amplified), "
              "reflectance (noisy where gain is high), the two branch outputs, the fusion weight of the deep branch and the final image."))

st.append(P("3.3 Ablation study", h2))
st.append(P("Each row removes or replaces exactly one component; dPSNR is the change against the full pipeline of the same family (negative = the component helps PSNR). "
            "Pooled over 30 held-out paired images from the three test sets.", small))
st.append(tbl(abl, [8.4 * cm, 1.9 * cm, 1.6 * cm, 1.7 * cm, 1.6 * cm, 2.2 * cm], align_center_from=1, font=7.8))
st.append(Spacer(1, 3))
r_nc = A.loc["hybrid - classical denoise"]; r_cl = A.loc["hybrid - CLAHE"]; r_dl = A.loc["hybrid - classical branch (= DL + post only)"]
r_ap = A.loc["hybrid - adaptive parameters (fixed)"]; r_cc = A.loc["hybrid - colour correction"]; r_h = A.loc["hybrid (full)"]
st.append(bullets([
    f"<b>Load-bearing components:</b> the gain-aware classical denoiser (without it SSIM falls from {r_h.SSIM:.2f} to {r_nc.SSIM:.2f} and NIQE rises from {r_h.NIQE:.1f} to {r_nc.NIQE:.1f}), "
    f"the Wiener shrinkage, our illumination decomposition (MSRCP instead: {A.loc['classical: MSRCP instead of illumination decomposition', 'PSNR'] - full_c:+.1f} dB) and the deep branch "
    f"({full_h - A.loc['hybrid - deep branch (= classical only)', 'PSNR']:+.1f} dB over classical alone). <b>Adaptive parameters</b> are worth {full_c - A.loc['classical - adaptive parameters (fixed)', 'PSNR']:+.1f} dB "
    f"(classical) and {full_h - r_ap.PSNR:+.1f} dB (hybrid) over fixed ones in PSNR.",
    f"<b>Honest negative results.</b> The hybrid <i>without</i> its classical branch (deep learning + post-processing only) scores slightly higher on this pool (PSNR {r_dl.PSNR - r_h.PSNR:+.2f} dB, "
    f"SSIM {r_dl.SSIM:.3f} vs {r_h.SSIM:.3f}), so the classical branch earns its place on very noisy / very dark regions and on over-exposure robustness rather than on the average LOL score. "
    f"Colour correction slightly <i>lowers</i> PSNR on LOL ({r_cc.PSNR - r_h.PSNR:+.2f} dB when removed), and CLAHE trades perceptual scores (SSIM {r_cl.SSIM:.3f}, NIQE {r_cl.NIQE:.2f} without it) for PSNR; "
    "the illumination prior, pyramid blending and output guard have no measurable effect on these uniformly dark images (their role is robustness on non-uniform and brighter inputs).",
    "<b>Image-quality-aware strategy.</b> On LOL the quality-aware wrapper equals raw Zero-DCE because every LOL image is very dark. On better-lit unseen images it cuts over-exposed pixels by "
    "about 30-60% and keeps the mean brightness near the target (e.g. VV: 2.0% vs 5.1% over-exposed pixels; NPE mean luma 0.40 vs 0.59; "
    "<font name=\"Courier\">results/quality_aware_check.md</font>)."]))

# 4 -------------------------------------------------------------------
st.append(P("4. Failure Modes &amp; Limitations", h1))
st.append(bullets([
    "<b>Observed failure case 1 - dark saturated colours and smoothing artefacts (Fig. 5).</b> <i>Root cause:</i> where the input chroma is close to the noise floor, colour has to be "
    "inferred; our conservative colour correction (strength capped at 0.6, gain range 0.75-1.33) limits both over-saturation and recovery, and the guided-filter / shrinkage stages smooth "
    "low-contrast areas into swirl patterns. <i>Mitigation:</i> a small learned chroma-restoration network on top of the fused luminance, and lowering the shrinkage strength where gain is moderate.",
    "<b>Observed failure case 2 - extremely dark input (LOL 55, Fig. 2 row 1, input mean luminance 0.04).</b> The hybrid is cleaner than either branch alone but fine speckle remains in the shadows and "
    "whites keep a green tint (11.5 dB). <i>Root cause:</i> gains reach the cap (about 16x), so the noise floor approaches the signal, and grey-world white balance only partly corrects a cast that "
    "varies across the image. <i>Mitigation:</i> a learned denoiser before the curves (e.g. a small U-Net trained on LOL) and spatially varying white balance.",
    "<b>Observed failure case 3 - strongly non-uniform lighting (four of the five lowest-PSNR images have non-uniformity 0.79-1.00).</b> <i>Root cause:</i> one smoothing radius for the illumination map "
    "cannot match both the light-source boundary and the shadow structure, so shadows are under-lifted. <i>Mitigation:</i> multi-scale / per-region illumination estimation.",
    "<b>Reference mismatch and untested cases.</b> The lowest-PSNR image overall (8.2 dB) is an already well-lit scene whose synthetic reference is washed out, so the output looks natural but scores low: a limit of "
    "full-reference scoring. Not observed: halos or measurable over-exposure (0.000 fraction on all three paired sets). Coloured illuminants (neon, sunset) were not tested.",
    "<b>Limitations and ethics.</b> Short training schedule (25 epochs on a laptop GPU), at most 30 test images per set, local NIQE model. Enhanced regions are plausible reconstructions, not ground truth: do not use "
    "for forensic, medical or evidentiary purposes, and disclose enhancement when sharing images."]))

st.append(fig("fig4_failure.jpg", 13.0,
              "Figure 5. Worst-case example (LOL-v2 Synthetic r03193652t, hybrid PSNR 10.5 dB): panels are input, input x4, classical, deep, hybrid, ground truth. "
              "Dark saturated petals stay dull brown and the green background shows swirl-like smoothing artefacts."))

# 5 -------------------------------------------------------------------
st.append(P("5. Deployment &amp; Quickstart Guide", h1))
st.append(P("<b>Quickstart commands</b>", body))
st.append(Preformatted(
    f"# 1. Clone the repository and install dependencies\n"
    f"git clone {REPO}.git\n"
    f"cd Computer_Vision_Low_Light_Image_Restoration\n"
    f"pip install -r requirements.txt\n\n"
    f"# 2. Run inference on a sample directory (weights/zero_dce.pt is in the repo)\n"
    f"python run_inference.py --input_dir ./samples/ --output_dir ./results/\n"
    f"#    optional: --method classical|deep|hybrid   --max_side 1200\n\n"
    f"# 3. Run the web app locally (http://localhost:8501)\n"
    f"streamlit run app/streamlit_app.py\n\n"
    f"# 4. Reproduce the evaluation (after placing the datasets as in data/README.md)\n"
    f"python scripts/evaluate.py --data data/LOL/eval15 data/LOL-v2/Real_captured/Test data/LOL-v2/Synthetic/Test \\\n"
    f"    --unpaired data/unseen/DICM data/unseen/LIME data/unseen/MEF data/unseen/NPE data/unseen/VV data/unseen/Fusion \\\n"
    f"    --max-images 30 --ablation --failures 5 --team VisionX --usns 1RUA24CSE0525,1RUA24CSE0544 --out results/final\n"
    f"python -m pytest -q          # 42 tests", code))
st.append(P("<b>Deployment format.</b> A single Streamlit application (<font name=\"Courier\">app/streamlit_app.py</font>) deployed on Streamlit Community Cloud from this repository "
            "(CPU only, no secrets): upload or generate a sample, choose Hybrid / Classical / Deep learning / Compare all three, toggle adaptive parameters, and inspect the result, intermediate "
            "pipeline steps, measured image statistics with the chosen parameters, histograms and metrics (PSNR/SSIM/LPIPS when a reference is supplied), then download the PNG. Corrupt, oversized "
            "or unsupported files produce a readable message; large photos are down-scaled while decoding. A <font name=\"Courier\">Dockerfile</font> (CPU PyTorch, honours "
            "$PORT) and a batch CLI (<font name=\"Courier\">run_inference.py</font>) provide the same pipeline in a container or from the command line. Typical latency is 0.1-0.6 s on a laptop CPU "
            "and about 2 s for an 800-px hybrid run on the free cloud CPU.", body))
st.append(KeepTogether([P("<b>Student submission checklist</b>", body), bullets([
    "[x] Working GitHub repository link with an updated README.md.",
    "[x] No raw code pasted into this report (modules are referenced by file name).",
    "[x] All table metrics come from results/final/*.csv and are reproducible with scripts/evaluate.py (latency is hardware-dependent).",
    "[x] All team member details recorded."])]))


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5); c.setFillColor(GREY)
    c.drawString(1.6 * cm, 1.0 * cm, f"Team {TEAM} - P1 Robust Low-Light Image Restoration and Enhancement")
    c.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {d.page}")
    c.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.7 * cm,
                        title="Hackathon Technical Solution Report - Team VisionX - P1", author=MEMBERS, subject="Robust Low-Light Image Restoration and Enhancement")
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
