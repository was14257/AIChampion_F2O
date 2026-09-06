import os
import secrets
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import gaussian_filter1d
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG
from bscan_gen.utils import load_retfound_encoder, embed_fundus
from bscan_gen import diffusion_bscan as D
from bscan_gen.ood_gate import Gate as OODGate
from glaucoma_cls.eyeon_cbm import EyeonCBM
from glaucoma_cls.concepts import ALL_CONCEPTS_V2, CONCEPT_META
from retfound_seg.model import build_model as build_seg_model
from glaucoma_cls.model import GlaucomaNet
from glaucoma_cls.data import _letterbox_square, _MEAN as _CLS_MEAN, _STD as _CLS_STD
import local_config as _lc
from local_config import CLS_UNFREEZE_LAST_N, CLS_CONCEPT_PROJ_DIM
from glaucoma_cls.explain import concept_saliency, attention_rollout, overlay_heatmap, mc_dropout_ci

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# v2: 11 concepts (SEG6+RNFL5) trained on GAMMA+REFUGE+GRAPE pool (stratified
# 5-fold) instead of the old 9-concept GAMMA-only setup.
GLAUCOMA_CKPT = CFG.paths.output_root / "glaucoma_cls" / "glaucoma_v2_11concept_bestfold.pth"
SEG_CKPT = CFG.paths.ckpt_dir / "best.pth"
CONCEPT_NPZ_V2 = CFG.paths.oct_features / "cbm_concepts_v2.npz"
RNFL_PLS_CKPT = CFG.paths.oct_features / "grape_rnfl_pls.pkl"
# Stage B diffusion: the 512 model (512 reproduces structure better than 256).
# Carrying resolution+checkpoint as one spec means we no longer patch
# diffusion_bscan's globals or keep a duplicate sampler here.
DIFF_SPEC = D.spec_512()

_models = {}


def _load_all():
    """Load all models once into a global cache so Streamlit doesn't reload per request."""
    if _models:
        return _models

    print("Loading models...")

    # --- OOD gate ---
    _models["ood_gate"] = OODGate()

    # --- Encoders for concept computation: segmentation + OCT-regression ---
    seg_model = build_seg_model(load_weights=True)
    seg_ck = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    seg_model.load_state_dict(seg_ck["model"])
    seg_model.eval().to(DEVICE)

    oct_enc = load_retfound_encoder()
    oct_enc.eval().to(DEVICE)

    import pickle
    with open(RNFL_PLS_CKPT, "rb") as f:
        rnfl_model = pickle.load(f)

    _models["cbm"] = EyeonCBM(seg_model, oct_enc, oct_linear=None,
                              rnfl_model=rnfl_model).to(DEVICE)

    # --- Glaucoma risk classifier (fused with 11 concepts, v2) ---
    concept_table, n_concepts = _load_concept_table_v2()
    clf = GlaucomaNet(freeze_encoder=True, unfreeze_last_n=CLS_UNFREEZE_LAST_N,
                      n_concepts=n_concepts, concept_proj_dim=CLS_CONCEPT_PROJ_DIM).to(DEVICE)
    ck = torch.load(GLAUCOMA_CKPT, map_location=DEVICE, weights_only=False)
    clf.load_state_dict(ck["model"])
    clf.eval()
    _models["clf"] = clf
    _models["concept_table"] = concept_table
    _models["concept_stats"] = _fit_concept_norm_v2(concept_table, n_concepts)
    _models["n_concepts"] = n_concepts

    # --- Stage A: fundus embedding -> thickness profile PLS (final-fit on 172 hand-labeled cases) ---
    _models["stageA"] = _fit_stage_a()

    # --- Stage B: diffusion UNet ---
    _models["unet"] = D.load_unet(DIFF_SPEC, DEVICE)

    print("Model loading complete")
    return _models


def _load_concept_table_v2():
    """Load cbm_concepts_v2.npz (11 concepts: SEG6+RNFL5, GAMMA_train+REFUGE+GRAPE, n=758)
    as a filename -> concept vector dict."""
    from pathlib import Path
    d = np.load(CONCEPT_NPZ_V2, allow_pickle=True)
    paths = d["paths"].tolist()
    X = d["X"].astype("float32")
    return {Path(p).name: X[i] for i, p in enumerate(paths)}, X.shape[1]


def _fit_concept_norm_v2(concept_table, n_concepts):
    """Reproduce the concept normalization stats used in v2 training
    (temp/cv_v2_11concept.py): full GAMMA_train+REFUGE+GRAPE pool (n=758),
    unified across folds for deployment."""
    X = np.stack(list(concept_table.values()))
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _fit_stage_a():
    """Reproduce Stage A from bscan_gen/fundus_to_oct_e2e.py: final-fit PLS on
    172 hand-labeled embeddings, also returning the 172 RPE shapes used for
    mean shape / personalization."""
    FEAT = CFG.paths.oct_features / "features.csv"
    EMB = CFG.paths.oct_features / "emb_whole_final.npz"
    LINES = CFG.paths.oct_labels / "lines"
    Wc = CFG.oct_tier1.worig

    df = pd.read_csv(FEAT, dtype={"case_id": str})
    df["case_id"] = df["case_id"].str.zfill(4)
    z = np.load(EMB, allow_pickle=True)
    X = z["X"]
    ids = df["case_id"].values

    ILM, RPE, keep = [], [], []
    for i, c in enumerate(ids):
        p = LINES / f"{c}.npz"
        if p.exists():
            d = np.load(p)
            if not (np.isnan(d["ilm"]).any() or np.isnan(d["rpe"]).any()):
                ILM.append(d["ilm"]); RPE.append(d["rpe"]); keep.append(i)
    ILM = np.stack(ILM); RPE = np.stack(RPE)
    Xk = X[keep]

    TH = RPE - ILM
    mean_th = TH.mean(0)
    rpe_shapes = RPE - RPE.mean(1, keepdims=True)
    base = float(np.median(RPE))
    x = np.arange(Wc)
    ec = np.abs(x - Wc / 2) / (Wc / 2)
    wgt = gaussian_filter1d(np.clip((ec - 0.08) / 0.17, 0, 1), 10)

    sc = StandardScaler().fit(Xk)
    pls = PLSRegression(8).fit(sc.transform(Xk), TH)

    return {"scaler": sc, "pls": pls, "mean_th": mean_th, "TH": TH,
            "rpe_shapes": rpe_shapes, "base": base, "wgt": wgt, "Wc": Wc}


def _build_ilm_rpe(TH_w, sA):
    """RPE shape (tilt etc.) can't be predicted from fundus, so borrow the real
    RPE shape from the hand-labeled case with the closest thickness profile
    (nearest-neighbor match, same as Stage A in fundus_to_oct_e2e.py) to add
    variety instead of a flat shape."""
    nn_idx = int(np.argmin(((TH_w[None, :] - sA["TH"]) ** 2).mean(axis=1)))
    rpe_g = sA["base"] + sA["rpe_shapes"][nn_idx]
    ilm_g = rpe_g - TH_w
    return ilm_g, rpe_g

st.set_page_config(
    page_title="EYEON",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Login credentials: read from env var, fall back to local_config.py (gitignored)
# so nothing plaintext ends up in the repo. Login is blocked if neither is set.
LOGIN_ID = os.environ.get("EYEON_LOGIN_ID") or getattr(_lc, "LOGIN_ID", None)
LOGIN_PASSWORD = os.environ.get("EYEON_LOGIN_PASSWORD") or getattr(_lc, "LOGIN_PASSWORD", None)

# Page styling (same CSS as the original app.py)
st.markdown(
    """
    <style>
        [data-testid="stHeader"] {
            background: rgba(255,255,255,0);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .eyeon-logo {
            font-size: 2.55rem;
            font-weight: 850;
            letter-spacing: -0.05em;
            margin-bottom: 0.15rem;
            color: #0f2b46;
        }

        .eyeon-subtitle {
            font-size: 1.05rem;
            color: #5b6875;
            margin-bottom: 1.8rem;
        }

        .login-space {
            height: 8vh;
        }

        .login-logo {
            text-align: center;
            font-size: 2.8rem;
            font-weight: 850;
            color: #0f2b46;
            letter-spacing: -0.05em;
            margin-bottom: 0.25rem;
        }

        .login-subtitle {
            text-align: center;
            color: #667482;
            margin-bottom: 1.6rem;
        }

        .section-card {
            border: 1px solid #dce4ec;
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            background: #ffffff;
            box-shadow: 0 7px 22px rgba(15, 43, 70, 0.05);
        }

        .result-card {
            border: 1px solid #dce4ec;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: #f8fbfd;
            margin-top: 0.8rem;
        }

        .footer-note {
            text-align: center;
            color: #7a8793;
            font-size: 0.84rem;
            padding-top: 0.8rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #dce4ec;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            background: #ffffff;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 700;
        }

        /* Patient-view result hero card */
        .hero-card {
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            margin: 0.4rem 0 1.1rem 0;
            color: #ffffff;
        }
        .hero-card .hero-label { font-size: 0.95rem; opacity: 0.9; }
        .hero-card .hero-title { font-size: 1.9rem; font-weight: 850; margin: 0.2rem 0; }
        .hero-card .hero-sub { font-size: 1rem; opacity: 0.95; line-height: 1.5; }
        .hero-green  { background: linear-gradient(135deg,#2e9e6b,#1f7a52); }
        .hero-amber  { background: linear-gradient(135deg,#e08a2b,#c56a12); }
        .hero-red    { background: linear-gradient(135deg,#d6483f,#b02a22); }

        /* Step-by-step guidance boxes */
        .step-box {
            border: 1px solid #dce4ec; border-radius: 14px;
            padding: 1rem 1.2rem; margin-bottom: 0.8rem; background: #ffffff;
        }
        .step-box .step-h { font-weight: 800; font-size: 1.05rem; color: #0f2b46; margin-bottom: 0.35rem; }
        .step-box .step-b { color: #3d4a57; line-height: 1.6; font-size: 0.96rem; }

        .gauge-track { background:#eef2f6; border-radius:999px; height:14px; width:100%; overflow:hidden; }
        .gauge-fill  { height:14px; border-radius:999px; }
        .disclaimer {
            border-left: 4px solid #b8c4d0; background:#f4f7fa; border-radius:8px;
            padding:0.7rem 1rem; color:#5b6875; font-size:0.86rem; line-height:1.5; margin-top:0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def login_page():
    st.markdown('<div class="login-space"></div>', unsafe_allow_html=True)

    left, center, right = st.columns([1.1, 1, 1.1])

    with center:
        st.markdown('<div class="login-logo">👁️ EYEON</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-subtitle">AI 기반 녹내장 조기 선별 시스템</div>',
            unsafe_allow_html=True,
        )

        if not LOGIN_ID or not LOGIN_PASSWORD:
            st.error(
                "로그인 정보가 설정되지 않았습니다. 환경변수 "
                "`EYEON_LOGIN_ID`/`EYEON_LOGIN_PASSWORD`를 지정하거나 "
                "`local_config.py`에 `LOGIN_ID`/`LOGIN_PASSWORD`를 추가하세요."
            )
            return

        with st.form("login_form", clear_on_submit=False):
            user_id = st.text_input("아이디", placeholder="아이디를 입력하세요")
            password = st.text_input(
                "비밀번호",
                type="password",
                placeholder="비밀번호를 입력하세요",
            )
            submitted = st.form_submit_button(
                "로그인",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            # compare_digest reduces timing-attack surface (the guard above
            # already ensures empty credentials never match).
            ok = (secrets.compare_digest(user_id, LOGIN_ID) and
                  secrets.compare_digest(password, LOGIN_PASSWORD))
            if ok:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

        st.caption("허가된 사용자만 접속할 수 있습니다.")


@torch.no_grad()
def run_analysis(img: Image.Image, progress_bar, eye_side: str = "auto"):
    """Run the full model pipeline (OOD gate -> 11 concepts -> risk score ->
    Stage A thickness prediction -> Stage B diffusion), updating the
    streamlit progress bar along the way.

    eye_side: "auto" (default, inferred from disc position) / "OD" / "OS".
    Stage A/B training data mixed OD/OS without left-right flipping, so
    nasal/temporal direction is inconsistent per case; RETFound embeddings
    are nearly flip-invariant so retraining with flips didn't fix it. Instead
    we generate as OD by default and flip the thickness profile/images for OS
    to match what the user actually sees.

    Auto-detection: the optic disc always sits nasal to the fovea (right of
    center for OD, left for OS), so disc-vs-fovea position perfectly
    separates laterality (100% on GAMMA n=200). No fovea detector here, so we
    approximate with image center (94%, 188/200 on the same set)."""
    from bscan_gen.utils import _retfound_tf

    m = _load_all()
    img = img.convert("RGB")

    progress_bar.progress(5, text="영상 품질 확인 중…")
    ood_ok, ood_dist = m["ood_gate"].check_image(img)
    if not ood_ok:
        return {"ood_reject": True, "ood_dist": ood_dist, "ood_thr": m["ood_gate"].thr}

    progress_bar.progress(20, text="시신경 구조 분석 중… (concept 11개 계산)")
    # predict_from_path needs a path, so save the uploaded image to a unique
    # temp file per request (a fixed filename previously let concurrent
    # requests overwrite each other) and clean up afterward.
    with tempfile.TemporaryDirectory(prefix="eyeon_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "input.png"
        img.save(tmp_path)
        cbm_out = m["cbm"].predict_from_path(str(tmp_path), device=DEVICE)
    concept_vec_raw = np.array([cbm_out[c] for c in ALL_CONCEPTS_V2], dtype=np.float32)
    mean, std = m["concept_stats"]
    concept_vec_norm = (concept_vec_raw - mean) / std
    seg_overlay, disc_x_rel = _disc_cup_overlay(m["cbm"], img)

    inferred_eye_side = _infer_eye_side(disc_x_rel)
    if eye_side == "auto":
        eye_side = inferred_eye_side or "OD"  # fall back to OD if detection fails

    progress_bar.progress(45, text="질환 위험도 산출 중…")
    # GlaucomaNet was retrained with aspect-ratio-preserving center-crop, unlike
    # the shared _retfound_tf() (plain Resize, no crop) used by Stage A/B and
    # concept extraction, so it needs its own preprocessing transform here.
    x = transforms.Compose([
        transforms.Lambda(_letterbox_square),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_CLS_MEAN, _CLS_STD),
    ])(img).unsqueeze(0).to(DEVICE)
    c_in = torch.from_numpy(concept_vec_norm).float().unsqueeze(0).to(DEVICE)
    risk_prob = torch.sigmoid(m["clf"](x, c_in)).item()

    # MC-dropout CI: 30 forward passes with dropout on give a prediction
    # distribution; a narrow interval means high confidence, wide means uncertain.
    progress_bar.progress(48, text="예측 신뢰구간 추정 중… (MC-dropout)")
    risk_ci = mc_dropout_ci(m["clf"], x, c_in, n=30)

    # --- Two post-hoc explainability paths: concept saliency + attention map ---
    # Attention Rollout is used over Grad-CAM since it localizes the optic disc
    # better for ViTs and has fewer background/border artifacts.
    # concept_saliency needs backward (enable_grad); rollout only needs no_grad.
    progress_bar.progress(52, text="판단 근거 분석 중… (concept 기여도 + attention)")
    with torch.enable_grad():
        saliency = concept_saliency(m["clf"], x, c_in, concept_names=ALL_CONCEPTS_V2)
    att_map, _ = attention_rollout(m["clf"], x, c_in)
    gradcam_overlay = np.asarray(overlay_heatmap(img.resize((224, 224)), att_map))

    progress_bar.progress(60, text="망막 두께 프로파일 예측 중…")
    whole_emb = embed_fundus(m["cbm"].oct_encoder, img)
    sA = m["stageA"]
    Xs = sA["scaler"].transform(whole_emb[None, :])
    TH_pred = sA["pls"].predict(Xs)[0]
    TH_w = sA["mean_th"] + sA["wgt"] * (TH_pred - sA["mean_th"])
    ilm, rpe = _build_ilm_rpe(TH_w, sA)

    # Generation defaults to OD (see docstring above); flip the thickness
    # profile for OS inputs so nasal/temporal direction matches the real eye
    # before passing it as the diffusion condition.
    if eye_side == "OS":
        ilm, rpe = ilm[::-1].copy(), rpe[::-1].copy()

    cond = D.build_cond_for(ilm, rpe, DIFF_SPEC)
    cond_t = torch.from_numpy(cond)[None, None].to(DEVICE)

    # DDIM (eta=0) is deterministic but the starting noise is random each time,
    # so speckle pattern varies across samples. We generate 5 and auto-pick the
    # smoothest (least speckle) one — this only picks the nicer-looking sample,
    # not a more accurate one (structure is identical across all 5, only
    # speckle differs).
    # Looping 5x sequentially would mean 500 UNet forward calls; instead we
    # repeat cond into a batch of 5 and sample once (100 forward calls, batch
    # size 5). Reduce N_SAMPLES if this OOMs on tighter VRAM budgets.
    N_SAMPLES = 5

    def _diff_progress(k, total):
        pct = 60 + int(35 * k / total)
        progress_bar.progress(min(pct, 95), text="합성 OCT 생성 중…")

    cond_batch = cond_t.repeat(N_SAMPLES, 1, 1, 1)
    gen = D.ddim_sample(m["unet"], cond_batch, DIFF_SPEC, steps=100,
                        progress_cb=_diff_progress)
    gen_imgs = ((gen.clamp(-1, 1) + 1) * 127.5).cpu().numpy()[:, 0].astype(np.uint8)
    candidates = [gen_imgs[i] for i in range(N_SAMPLES)]

    def _speckle_score(im):
        """High-frequency (speckle) energy — lower means smoother/nicer looking."""
        from scipy.ndimage import gaussian_filter
        f = im.astype(np.float32)
        hf = f - gaussian_filter(f, sigma=1.5)
        return float(hf.std())

    gen_img = min(candidates, key=_speckle_score)

    progress_bar.progress(100, text="분석 완료")
    time.sleep(0.2)
    progress_bar.empty()

    concept_dict = dict(zip(ALL_CONCEPTS_V2, concept_vec_raw.tolist()))
    return {
        "ood_reject": False,
        "ood_dist": ood_dist,
        "risk_prob": risk_prob,
        "risk_ci": risk_ci,
        "concept": concept_dict,
        "saliency": saliency,
        "gradcam_overlay": gradcam_overlay,
        "eye_side": eye_side,
        "eye_side_auto_detected": inferred_eye_side is not None,
        "oct_image": gen_img,
        "seg_overlay": seg_overlay,
    }


def _saliency_figure(saliency: dict):
    """Horizontal bar chart of concept saliency (gradient*input), sorted by
    absolute magnitude. Red=risk-increasing, blue=protective. Uses Malgun
    Gothic so CONCEPT_META's Korean labels render correctly."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    names = list(saliency.keys())
    vals = np.array([saliency[n] for n in names])
    order = np.argsort(np.abs(vals))[::-1]
    names = [names[j] for j in order]
    vals = vals[order]
    labels = [CONCEPT_META.get(n, (n,))[0] for n in names]
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in vals]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    y = range(len(vals))[::-1]
    ax.barh(list(y), vals, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("위험도 기여도 (빨강=위험 ↑ / 파랑=위험 ↓)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _disc_cup_overlay(cbm, img: Image.Image) -> tuple:
    """Segment disc/cup with cbm's seg_model and overlay them on the original
    fundus image (shows the actual C/D segmentation instead of a condition
    sketch). Follows retfound_seg.cdr.postprocess_label's label convention
    (0=background, disc_rim=disc, cup=cup), matching
    glaucoma_cls.eyeon_cbm.EyeonCBM.predict_from_path's pre/post-processing.

    Also returns the disc center's relative x-position (0-1) for OD/OS
    auto-detection (see _infer_eye_side)."""
    from config import CFG
    from glaucoma_cls.eyeon_cbm import _preprocess_from_image
    from retfound_seg.cdr import postprocess_label, masks_from_label

    seg_input = _preprocess_from_image(img, DEVICE)
    logits = cbm.seg_model(seg_input)
    label_map = postprocess_label(logits.argmax(dim=1)[0].cpu().numpy())
    disc, cup = masks_from_label(label_map)

    size = CFG.data.img_size
    base = np.asarray(img.resize((size, size), Image.BILINEAR).convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    # Disc outline in green, cup outline in red (semi-transparent fill)
    disc_fill = np.zeros_like(base); disc_fill[disc] = [46, 204, 113]
    cup_fill = np.zeros_like(base); cup_fill[cup] = [217, 54, 62]
    overlay = np.where(disc[..., None], overlay * 0.55 + disc_fill * 0.45, overlay)
    overlay = np.where(cup[..., None], overlay * 0.55 + cup_fill * 0.45, overlay)

    disc_cols = np.where(disc.any(axis=0))[0]
    disc_x_rel = float(disc_cols.mean() / disc.shape[1]) if len(disc_cols) else float("nan")

    return overlay.astype(np.uint8), disc_x_rel


def _infer_eye_side(disc_x_rel: float) -> str | None:
    """Classify OD if the disc center is right of image center, OS if left.
    Validated at 94% (188/200) on GAMMA laterality.csv using image-center as a
    fovea proxy (no fovea detector available; fovea-relative position would
    give 100% but we don't have that). Returns None if disc wasn't detected
    or sits too close to center to trust."""
    if disc_x_rel != disc_x_rel:  # NaN
        return None
    if abs(disc_x_rel - 0.5) < 0.03:  # too close to center to trust
        return None
    return "OD" if disc_x_rel > 0.5 else "OS"


def _confidence_from_ci(ci: dict) -> tuple:
    """Grade confidence by MC-dropout CI width — narrower interval means
    more stable prediction, hence higher confidence."""
    width = ci["hi"] - ci["lo"]
    if width < 0.10:
        return "높음", "🟢"
    if width < 0.25:
        return "보통", "🟡"
    return "낮음", "🔴"


# Decision thresholds (on raw risk). Kept in one place so screen/PDF stay in
# sync (previously report_pdf.py hardcoded a stale value separately).
# Recalibrated for v2 (11 concepts incl. RNFL, GAMMA+REFUGE+GRAPE pool) using
# out-of-fold probabilities (oof_v2_11concept.npz, n=763, auc=0.9605, no
# leakage). Requiring sens=1.0 (v1 policy) collapses spec to 0 on this wider
# pool, so the floor is relaxed to the last point keeping sens>=0.95.
THR_SUSPECT = 0.170  # "needs attention" entry point (last point with sens>=0.95; OOF sens=0.958/spec=0.749)
THR_HIGH = 0.861     # "high risk" entry point (max Youden's J; OOF sens=0.858/spec=0.963)


def _risk_grade(risk_prob):
    """Map risk -> (grade name, hero CSS class, headline, detail, recommended action).

    Thresholds are recalibrated per retrain on true held-out data (currently
    REFUGE val, n=399). Lower bound = last point keeping sens=1.0 (FN=0,
    screening priority); upper bound = max Youden's J (sens+spec-1). Must be
    recomputed whenever the model is retrained, and the concept normalization
    stats used must exactly match train.py's build_frames() call args —
    a stats mismatch previously shifted thresholds significantly."""
    if risk_prob >= THR_HIGH:
        return ("높은 위험", "hero-red",
                "녹내장이 강하게 의심됩니다",
                "AI 분석 결과 녹내장과 관련된 시신경 변화가 뚜렷하게 관찰되었습니다. "
                "녹내장은 초기에 자각 증상이 거의 없지만, 방치하면 서서히 시야가 좁아지고 "
                "회복이 어려운 시력 손상으로 이어질 수 있습니다.",
                "가능한 한 빨리 안과를 방문하여 OCT 등 정밀 검사를 받아보시길 권합니다.")
    if risk_prob >= THR_SUSPECT:
        return ("주의 필요", "hero-amber",
                "녹내장 가능성이 있어 확인이 필요합니다",
                "AI 분석에서 녹내장을 의심할 만한 소견이 일부 관찰되었습니다. 확정 진단은 "
                "아니지만, 초기 녹내장은 놓치면 위험하기 때문에 선별 단계에서는 의심 기준을 "
                "다소 넓게 잡아 안내드립니다.",
                "안과에 방문하여 정밀 검사로 실제 녹내장 여부를 확인하시길 권합니다.")
    return ("낮은 위험", "hero-green",
            "현재는 녹내장 위험이 낮습니다",
            "AI 분석 결과 녹내장과 관련된 뚜렷한 이상 소견은 관찰되지 않았습니다. 다만 본 "
            "결과는 선별 목적의 참고 자료이며, 녹내장은 서서히 진행하므로 정기적인 안과 "
            "검진을 유지하는 것이 중요합니다.",
            "특별한 증상이 없다면 1년에 한 번 정기 안과 검진을 권합니다.")


def _gauge_html(cls, color):
    """Fill the gauge to a fixed position per grade (not raw risk %) —
    uncalibrated probabilities cluster too tightly to be meaningful as a bar
    length, so the patient view only visualizes the final grade."""
    pct = {"hero-green": 25, "hero-amber": 60, "hero-red": 90}.get(cls, 50)
    return (f'<div class="gauge-track"><div class="gauge-fill" '
            f'style="width:{pct}%;background:{color};"></div></div>')


def _step(title, body):
    st.markdown(f'<div class="step-box"><div class="step-h">{title}</div>'
                f'<div class="step-b">{body}</div></div>', unsafe_allow_html=True)


DISCLAIMER = ("본 결과는 인공지능 기반 <b>선별(screening) 보조 도구</b>의 분석이며, 의사의 진단을 "
              "대체하지 않습니다. 최종 진단과 치료는 반드시 안과 전문의의 진료와 정밀 검사(OCT 등)를 "
              "통해 이루어져야 합니다. 학습 데이터가 <b>동아시아인 안저 사진</b>으로 구성되어 있어, "
              "다른 인종/지역에서는 정확도가 달라질 수 있습니다.")


def _render_patient_view(result):
    """Patient view: result -> meaning -> plain-language rationale -> next steps."""
    risk = result["risk_prob"]; ci = result["risk_ci"]; concept = result["concept"]
    cdr = concept.get("cdr", float("nan"))
    grade, cls, headline, detail, action = _risk_grade(risk)
    color = {"hero-green": "#2e9e6b", "hero-amber": "#e08a2b", "hero-red": "#d6483f"}[cls]

    # Hero card (only grade shown; raw risk % is uncalibrated and misleading)
    st.markdown(
        f'<div class="hero-card {cls}">'
        f'<div class="hero-label">AI 녹내장 위험도 선별 결과</div>'
        f'<div class="hero-title">{grade}</div>'
        f'<div class="hero-sub">{headline}</div></div>',
        unsafe_allow_html=True)

    # Risk gauge (by grade) + one-line confidence note
    st.markdown(_gauge_html(cls, color), unsafe_allow_html=True)
    conf_txt, conf_icon = _confidence_from_ci(ci)
    st.caption(f"AI 예측 신뢰도 {conf_icon} {conf_txt} "
               f"(비슷한 사진을 반복 분석했을 때 결과가 {'거의 일정' if conf_txt=='높음' else '다소 변동'}함)")

    # What does this mean
    _step("이 결과는 무슨 의미인가요?", detail)

    # Why this result (plain-language rationale)
    sal = result["saliency"]
    top = max(sal, key=lambda k: abs(sal[k]))
    top_label = CONCEPT_META.get(top, (top,))[0]
    cdr_txt = (f"시신경의 함몰비(C/D 비율)가 {cdr:.2f}로 "
               f"{'다소 높은 편입니다. 이 값이 클수록 녹내장 가능성이 올라갑니다.' if cdr>=0.6 else '정상 범위입니다.'}")
    _step("왜 이런 결과가 나왔나요?",
          f"AI는 시신경유두(눈 안쪽에서 신경이 모이는 부위)의 모양과 망막 구조를 종합해 "
          f"판단했습니다. 이번 분석에서 가장 크게 영향을 준 요소는 <b>{top_label}</b>였고, {cdr_txt} "
          f"오른쪽 ‘상세 분석’ 탭에서 AI가 주목한 부위를 이미지로 확인할 수 있습니다.")

    # What to do next
    _step("이제 어떻게 해야 하나요?", action)

    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


_PLATT_A, _PLATT_B = 1.844, 0.554  # Refit on REFUGE val (n=399, true held-out)
                                   # with LogisticRegression C=0.1 (strong reg.) —
                                   # weaker reg. (e.g. C=10) overfit the small
                                   # held-out positive set (n=40), pushing
                                   # normal-group max probability up to 75%.


def _platt_scale(p):
    """Display-only probability recalibration (Platt scaling). Raw sigmoid
    outputs cluster tightly for both classes, making everything look low.
    Single-parameter temperature scaling barely helped (model isn't overconfident,
    just genuinely uncertain), so 2-param Platt (a*logit+b) spreads the
    distribution instead. Used ONLY for clinician-view display — decision
    thresholds stay on raw probability so validated sens/spec figures stay correct."""
    eps = 1e-7
    pc = min(max(p, eps), 1 - eps)
    z = _PLATT_A * np.log(pc / (1 - pc)) + _PLATT_B
    return 1 / (1 + np.exp(-z))


def _render_clinician_view(result):
    """Clinician view: detailed metrics, evidence, and XAI."""
    risk = result["risk_prob"]; ci = result["risk_ci"]; concept = result["concept"]
    cdr = concept.get("cdr", float("nan"))
    conf_txt, conf_icon = _confidence_from_ci(ci)
    risk_disp = _platt_scale(risk)  # recalibrated for display; decisions use raw risk

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("녹내장 위험도", f"{risk_disp*100:.1f}%",
              delta=f"의심 (≥{THR_SUSPECT*100:.0f}%)" if risk >= THR_SUSPECT else "저위험",
              delta_color="inverse",
              help="판정 기준 raw risk≥60%(재보정 전 확률, REFUGE val held-out "
                   "기준 sens=1.0 유지 상한점, 2026-07-30 crop 전처리 재학습 + 정규화 "
                   "통계 수정 후 재조정). 선별 목적상 위음성 최소화를 위해 보수적으로 "
                   "적용. 표시된 %는 확률 재보정(Platt scaling) 후 값으로 판정 threshold와 "
                   "직접 대응하지 않음.")
    k2.metric("95% 신뢰구간", f"{_platt_scale(ci['lo'])*100:.0f}–{_platt_scale(ci['hi'])*100:.0f}%",
              help="MC-Dropout 30회 추론 분포(재보정 표시).")
    k3.metric("수직 C/D 비율", f"{cdr:.3f}",
              delta="정상범위 초과" if cdr >= 0.6 else "정상범위", delta_color="inverse",
              help="Cup-to-Disc Ratio. 통상 0.6 이상 녹내장 의심.")
    k4.metric("예측 신뢰도", f"{conf_icon} {conf_txt}")

    # XAI, two columns
    st.markdown("###### 판단 근거 (XAI)")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.image(result["gradcam_overlay"],
                 caption="Attention Rollout — 모델 주목 영역(참고용, 신뢰도 낮음)",
                 use_container_width=True, clamp=True)
    with ex2:
        st.pyplot(_saliency_figure(result["saliency"]), use_container_width=True)
        st.caption("Concept saliency (gradient×input): 빨강=위험 기여, 파랑=보호 기여 "
                   "— 공간적 히트맵과 달리 이 수치 기반 근거는 별도 검증 대상 아님. "
                   "기준이 다름 주의: 위 표의 '정상범위'는 값 자체(절대치)를 IQR과 비교하지만, "
                   "이 그래프는 학습 데이터 평균 대비 표준화 점수(z-score)의 기여도라 "
                   "값이 정상범위를 벗어나도 z-score가 평균에 가까우면(=이 케이스가 유별나게 "
                   "낮은 편은 아니면) 기여도가 작게(약한 색으로) 나올 수 있음 — 서로 다른 "
                   "잣대라 방향이 항상 일치하지는 않음")

    # Structural images, two columns
    st.markdown("###### 합성 OCT · 시신경 구조")
    oc1, oc2 = st.columns(2)
    oc1.image(result["oct_image"], caption="합성 OCT B-scan (512×512)",
              use_container_width=True, clamp=True)
    oc2.image(result["seg_overlay"], caption="Disc(초록)/Cup(빨강) Segmentation",
              use_container_width=True, clamp=True)

    # Table of the 11 concepts
    st.markdown("###### 시신경/망막 정량 지표 (11종)")
    rows = {"지표": [], "값": [], "정상범위": [], "판정": [], "단위": [], "설명": []}
    for c in ALL_CONCEPTS_V2:
        v = concept.get(c, float("nan"))
        label, unit, desc, rng, direction = CONCEPT_META.get(c, (c, "", "", None, "high"))
        fmt = (lambda x: f"{x:,.0f}") if unit == "px" else (lambda x: f"{x:.3f}")
        rows["지표"].append(label)
        rows["값"].append(fmt(v))
        if rng is not None:
            lo, hi = rng
            rows["정상범위"].append(f"{fmt(lo)}~{fmt(hi)}")
            if v != v:  # NaN
                verdict = "—"
            elif lo <= v <= hi:
                verdict = "🟢 정상범위"
            elif (direction == "high" and v > hi) or (direction == "low" and v < lo):
                verdict = "🔴 주의"
            else:
                # Outside normal range but on the non-risk side (e.g. thicker
                # than average) — not a risk signal, so mark as normal.
                verdict = "🟢 정상범위"
        else:
            rows["정상범위"].append("—")
            verdict = "—"
        rows["판정"].append(verdict)
        rows["단위"].append(unit if unit else "—")
        rows["설명"].append(desc)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("정상범위는 GAMMA_train+REFUGE+GRAPE(n=758) 중 정상 판정군의 IQR(25~75%ile). "
               "RNFL Mean/I/S/N/T는 GRAPE(n=244 실측 OCT RNFL) 학습 회귀로 fundus에서 예측한 값(실측 아님). "
               "면적(px)은 512×512 마스크 픽셀 수로 상대 비교용. "
               "성능: AUROC 0.963±0.018 (GAMMA+REFUGE+GRAPE pool stratified 5-fold, n=758). 모델: glaucoma_cls v2 "
               f"(unf{CLS_UNFREEZE_LAST_N}+concept11+concept_proj{CLS_CONCEPT_PROJ_DIM}"
               "+pos_weight0.6). 선별 보조용, 확정 진단 아님.")


def _render_result(result):
    """Render the analysis result full-width; just shows a hint if result is None."""
    if result is None:
        st.info("왼쪽에서 안저사진을 업로드하고 'AI 분석 실행'을 눌러 주세요.")
        return

    if result.get("ood_reject"):
        st.error("✗ 안저사진으로 인식되지 않았습니다. 다른 사진을 업로드해 주세요.")
        return

    eye_label = "우안 (OD)" if result.get("eye_side") == "OD" else "좌안 (OS)"
    if result.get("eye_side_auto_detected"):
        st.caption(f"👁️ 촬영 부위 자동 판별: **{eye_label}** "
                    "(시신경유두 위치 기반, 잘못됐다면 왼쪽에서 직접 지정 후 재분석하세요)")
    else:
        st.caption(f"👁️ 촬영 부위: **{eye_label}** (자동 판별 실패로 기본값 적용 — "
                    "왼쪽에서 직접 지정 후 재분석을 권장합니다)")

    # PDF report download (cached to avoid regenerating for the same result)
    # No patient identity integration yet, so a placeholder name is used
    # (TODO: add a real name input to the upload form).
    from report_pdf import build_report_pdf
    if st.session_state.get("_pdf_cache_id") != id(result):
        st.session_state["_pdf_bytes"] = build_report_pdf(
            result, _risk_grade, _platt_scale,
            patient_name="홍길동 (예시)", confidence_fn=_confidence_from_ci,
            thr_suspect=THR_SUSPECT)
        st.session_state["_pdf_cache_id"] = id(result)
    pdf_bytes = st.session_state["_pdf_bytes"]
    st.download_button(
        "📄 PDF 리포트 다운로드", data=pdf_bytes,
        file_name="EYEON_결과리포트.pdf", mime="application/pdf",
        use_container_width=False,
    )

    # Separate patient-view / clinician-view tabs
    tab_p, tab_c = st.tabs(["🧑 환자용 결과", "🩺 의료진용 상세 분석"])
    with tab_p:
        _render_patient_view(result)
    with tab_c:
        _render_clinician_view(result)


def main_page():
    top_left, top_right = st.columns([5, 1])

    with top_left:
        st.markdown('<div class="eyeon-logo">👁️ EYEON</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="eyeon-subtitle">안저사진 기반 녹내장 조기 선별 시스템</div>',
            unsafe_allow_html=True,
        )

    with top_right:
        if st.button("로그아웃", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Upload (left) + preview (right); result renders full-width below
    up_col, prev_col = st.columns([1, 1], gap="large")
    with up_col:
        st.subheader("안저사진 업로드")
        uploaded_file = st.file_uploader(
            "JPG 또는 PNG 파일을 선택하세요.",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        eye_side_label = st.selectbox(
            "촬영 부위", ["자동 판별(권장)", "우안 (OD)", "좌안 (OS)"],
            help="합성 OCT의 코쪽/관자놀이쪽 방향을 맞추기 위한 정보입니다. "
                 "기본값은 시신경유두 위치로 자동 판별하며(정확도 약 94%), "
                 "필요 시 직접 지정할 수 있습니다.",
        )
        if eye_side_label.startswith("자동"):
            eye_side = "auto"
        elif eye_side_label.startswith("우안"):
            eye_side = "OD"
        else:
            eye_side = "OS"
        analyze = st.button(
            "AI 분석 실행", type="primary", use_container_width=True,
            disabled=uploaded_file is None,
        )

    valid_image = None
    if uploaded_file is not None:
        try:
            valid_image = Image.open(uploaded_file).convert("RGB")
            with prev_col:
                st.image(valid_image, caption="업로드된 안저사진", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("이미지 파일을 확인해 주세요.")

    if analyze and valid_image is not None:
        progress = st.progress(0, text="영상 품질 확인 중…")
        with st.spinner("모델 추론 중... (최초 1회는 다소 걸릴 수 있습니다)"):
            result = run_analysis(valid_image, progress, eye_side=eye_side)
        st.session_state["result"] = result

    if st.session_state.get("result") is not None:
        st.divider()
        st.subheader("분석 결과")
        _render_result(st.session_state["result"])
        st.caption("본 시스템의 결과는 의료진의 진단 및 실제 OCT 검사를 대체하지 않습니다.")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_page()
else:
    login_page()
