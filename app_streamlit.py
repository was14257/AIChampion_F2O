"""EYEON Streamlit 앱 — Downloads/EYEON/EYEON_clean_start/app.py의 로그인 화면과
디자인(카드 스타일, metric 레이아웃)을 그대로 재현하되, 고정 샘플 결과 대신
실제 모델 파이프라인(OOD gate, glaucoma risk, concept 9개, Stage A/B 합성 OCT)을
붙인 버전. 원본 app.py는 그대로 두고 이 파일만 새로 추가.

모델 로더/파이프라인은 원래 별도 Gradio 데모(e2e_demo.py)에 있었으나, Gradio UI
자체는 실제로 안 쓰이고 이 Streamlit 앱만 운영되어(2026-07-27) e2e_demo.py를
없애고 로더/유틸 부분만 이 파일로 병합했다.
"""
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
from glaucoma_cls.concepts import ALL_CONCEPTS, OCT_CONCEPTS, CONCEPT_META
from retfound_seg.model import build_model as build_seg_model
from glaucoma_cls.model import GlaucomaNet
from glaucoma_cls.data import (load_concept_table, build_frames, _letterbox_square,
                               _MEAN as _CLS_MEAN, _STD as _CLS_STD)
import local_config as _lc
from local_config import (CLS_UNFREEZE_LAST_N, CLS_CONCEPT_PROJ_DIM,
                          CLS_POS_WEIGHT_SCALE)
from glaucoma_cls.explain import concept_saliency, attention_rollout, overlay_heatmap, mc_dropout_ci

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# train.py가 저장하는 위치(OUTPUT_ROOT/glaucoma_cls/best.pth)와 같은 곳을 가리킨다.
# 예전엔 로컬 절대경로가 박혀 있어 서버에서 그대로 실행되지 않았다.
GLAUCOMA_CKPT = CFG.paths.output_root / "glaucoma_cls" / "best.pth"
SEG_CKPT = CFG.paths.ckpt_dir / "best.pth"
OCT_LINEAR_CKPT = CFG.paths.oct_features / "oct_linear.pt"
# Stage B diffusion: the 512 model (512 reproduces structure better than 256).
# Carrying resolution+checkpoint as one spec means we no longer patch
# diffusion_bscan's globals or keep a duplicate sampler here.
DIFF_SPEC = D.spec_512()

_models = {}


def _load_all():
    """모든 모델을 한 번만 로드해서 전역 캐시에 담는다(Streamlit이 요청마다
    새로 만들지 않도록)."""
    if _models:
        return _models

    print("모델 로딩 중...")

    # --- OOD gate ---
    _models["ood_gate"] = OODGate()

    # --- concept 계산용: segmentation + OCT-regression 인코더 ---
    seg_model = build_seg_model(load_weights=True)
    seg_ck = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    seg_model.load_state_dict(seg_ck["model"])
    seg_model.eval().to(DEVICE)

    oct_enc = load_retfound_encoder()
    oct_enc.eval().to(DEVICE)

    oct_ck = torch.load(OCT_LINEAR_CKPT, map_location="cpu", weights_only=False)
    oct_linear = nn.Linear(oct_ck["in_dim"], len(OCT_CONCEPTS))
    oct_linear.load_state_dict(oct_ck["state_dict"])
    oct_linear.eval().to(DEVICE)

    _models["cbm"] = EyeonCBM(seg_model, oct_enc, oct_linear).to(DEVICE)

    # --- glaucoma risk 분류기 (concept 9개 결합, best 구성) ---
    concept_table, n_concepts = load_concept_table()
    clf = GlaucomaNet(freeze_encoder=True, unfreeze_last_n=CLS_UNFREEZE_LAST_N,
                      n_concepts=n_concepts, concept_proj_dim=CLS_CONCEPT_PROJ_DIM).to(DEVICE)
    ck = torch.load(GLAUCOMA_CKPT, map_location=DEVICE, weights_only=False)
    clf.load_state_dict(ck["model"])
    clf.eval()
    _models["clf"] = clf
    _models["concept_table"] = concept_table
    _models["concept_stats"] = _fit_concept_norm(concept_table, n_concepts)
    _models["n_concepts"] = n_concepts

    # --- Stage A: fundus 임베딩 -> 두께 프로파일 PLS (172개 손라벨로 최종 fit) ---
    _models["stageA"] = _fit_stage_a()

    # --- Stage B: diffusion UNet ---
    _models["unet"] = D.load_unet(DIFF_SPEC, DEVICE)

    print("모델 로딩 완료")
    return _models


def _fit_concept_norm(concept_table, n_concepts):
    """glaucoma_cls 학습 때(train.py)와 정확히 동일한 표본으로 concept 정규화
    mean/std를 재현한다. train.py는 build_frames(use_datasets=("REFUGE",))로
    REFUGE(400)+GAMMA_train(80)=480장만 ext로 쓰는데, 이전엔 concept_table
    전체(캐시된 REFUGE+ORIGA+G1020+GAMMA 2127장)로 근사했었다 - concept마다
    스케일이 크게 달라(예: disc_area가 수천대) 이 표본 차이가 확률 median을
    0.45~0.56 수준으로 흔들 만큼 커서(2026-07-30, REFUGE val 재평가 중 발견)
    "근사해도 안정적"이라는 원래 가정이 틀렸음이 확인됐다. 반드시 학습과
    동일한 480장 표본을 써야 함."""
    from pathlib import Path
    ext_all, _, _ = build_frames(use_datasets=("REFUGE",))
    X = np.stack([concept_table[Path(p).name] for p in ext_all["path"]
                 if Path(p).name in concept_table])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _fit_stage_a():
    """bscan_gen/fundus_to_oct_e2e.py의 Stage A를 그대로 재현: 172개 손라벨
    임베딩으로 PLS를 최종 fit하고, RPE 평균 모양/개인화용 172개 RPE shape도
    같이 반환한다."""
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
    """RPE 모양(tilt 등)은 fundus로 예측 불가(CLAUDE.md 4절, skill~0.1)라, 예측된
    두께 프로파일(TH_w)과 가장 비슷한 손라벨 172개 케이스의 실제 RPE 모양을
    빌려와 다양성을 준다 (12절 "다 똑같아 보임" 개선 로직, fundus_to_oct_e2e.py
    Stage A와 동일한 최근접 매칭)."""
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

# 로그인 정보. 저장소에 평문으로 남기지 않으려고 환경변수 -> local_config.py
# (.gitignore 대상) 순으로 읽는다. 둘 다 없으면 로그인 자체를 막는다 - 기본
# 비밀번호를 코드에 박아두면 그게 그대로 배포로 나가기 때문.
LOGIN_ID = os.environ.get("EYEON_LOGIN_ID") or getattr(_lc, "LOGIN_ID", None)
LOGIN_PASSWORD = os.environ.get("EYEON_LOGIN_PASSWORD") or getattr(_lc, "LOGIN_PASSWORD", None)

# 화면 스타일 (원본 app.py의 CSS 그대로)
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

        /* 환자용 결과 히어로 카드 */
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

        /* 단계별 안내 스텝 */
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
            # compare_digest로 타이밍 공격 여지를 줄인다(입력이 비어도
            # 위 가드 때문에 빈 자격증명과 일치하는 일은 없음).
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
    """모델 파이프라인(OOD gate -> concept 9개 -> risk score -> Stage A 두께
    예측 -> Stage B diffusion) 전체를 실행. streamlit progress bar를 그때그때
    갱신한다.

    eye_side: "auto"(기본, disc segmentation으로 자동 판별) / "OD"(우안) /
    "OS"(좌안). Stage A/B 학습 데이터(GAMMA 172개 손라벨, laterality.csv
    기준 OD 119 / OS 81)가 좌우 flip 없이 섞인 채로 학습되어 nasal/temporal
    방향이 case마다 뒤바뀐 상태다. flip 재학습으로 해결을 시도했으나 오히려
    악화됨을 확인(CLAUDE.md 미결 항목 참고, RETFound 임베딩이 좌우 flip에
    거의 불변이라 근본 해결 불가) - 재학습 대신 표시 단계에서 OD 기준으로
    생성한 뒤 OS면 두께 profile과 이미지 결과물을 좌우 반전해 nasal/temporal
    방향을 사용자가 실제로 보는 눈과 맞춘다.

    auto 판별 근거: disc(시신경유두)는 항상 fovea 기준 nasal(코쪽) 방향에
    있고 OD는 nasal이 오른쪽/OS는 왼쪽이므로, disc가 fovea보다 어느 쪽에
    있는지로 laterality가 완벽히 갈린다(GAMMA 200장 검증 100%). 이 데모는
    fovea 검출기가 없어 대신 '이미지 정중앙'을 근사치로 쓴다(표준 촬영에서
    fovea가 대체로 중앙 근처에 옴) - 같은 200장 기준 정확도 94%(188/200)."""
    from bscan_gen.utils import _retfound_tf

    m = _load_all()
    img = img.convert("RGB")

    progress_bar.progress(5, text="영상 품질 확인 중…")
    ood_ok, ood_dist = m["ood_gate"].check_image(img)
    if not ood_ok:
        return {"ood_reject": True, "ood_dist": ood_dist, "ood_thr": m["ood_gate"].thr}

    progress_bar.progress(20, text="시신경 구조 분석 중… (concept 9개 계산)")
    # predict_from_path가 경로를 요구해서 업로드 이미지를 임시 저장한다. 예전엔
    # 고정 파일명이라 동시 요청 시 서로 덮어쓸 수 있었어서 요청마다 고유 파일을
    # 만들고 끝나면 지운다.
    with tempfile.TemporaryDirectory(prefix="eyeon_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "input.png"
        img.save(tmp_path)
        cbm_out = m["cbm"].predict_from_path(str(tmp_path), device=DEVICE)
    concept_vec_raw = np.array([cbm_out[c] for c in ALL_CONCEPTS], dtype=np.float32)
    mean, std = m["concept_stats"]
    concept_vec_norm = (concept_vec_raw - mean) / std
    seg_overlay, disc_x_rel = _disc_cup_overlay(m["cbm"], img)

    inferred_eye_side = _infer_eye_side(disc_x_rel)
    if eye_side == "auto":
        eye_side = inferred_eye_side or "OD"  # 판별 실패 시 기존 기본값(OD)으로

    progress_bar.progress(45, text="질환 위험도 산출 중…")
    # glaucoma_cls(GlaucomaNet)는 2026-07-30부터 종횡비 보존 center-crop(24절)으로
    # 재학습됨 - Stage A/B/concept 추출이 쓰는 공용 _retfound_tf()(crop 없이 그냥
    # Resize)를 그대로 쓰면 학습/추론 전처리가 어긋나므로 clf 전용 transform을 쓴다.
    x = transforms.Compose([
        transforms.Lambda(_letterbox_square),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_CLS_MEAN, _CLS_STD),
    ])(img).unsqueeze(0).to(DEVICE)
    c_in = torch.from_numpy(concept_vec_norm).float().unsqueeze(0).to(DEVICE)
    risk_prob = torch.sigmoid(m["clf"](x, c_in)).item()

    # MC-dropout 신뢰구간: dropout을 켠 채 30회 forward한 예측 분포로 CI 산출.
    # 구간이 좁으면 확신 높음, 넓으면 불확실 (이미지 특징 경로의 불확실성).
    progress_bar.progress(48, text="예측 신뢰구간 추정 중… (MC-dropout)")
    risk_ci = mc_dropout_ci(m["clf"], x, c_in, n=30)

    # --- 설명가능성 2경로 (post-hoc): concept 기여도 + attention 지도 ---
    # attention map은 Grad-CAM보다 disc(시신경유두)를 잘 짚는 Attention Rollout을
    # 쓴다 (ViT엔 rollout이 더 자연스럽고, 배경/테두리 아티팩트가 적음).
    # concept_saliency는 backward가 필요해 enable_grad, rollout은 no_grad로 충분.
    progress_bar.progress(52, text="판단 근거 분석 중… (concept 기여도 + attention)")
    with torch.enable_grad():
        saliency = concept_saliency(m["clf"], x, c_in)
    att_map, _ = attention_rollout(m["clf"], x, c_in)
    gradcam_overlay = np.asarray(overlay_heatmap(img.resize((224, 224)), att_map))

    progress_bar.progress(60, text="망막 두께 프로파일 예측 중…")
    whole_emb = embed_fundus(m["cbm"].oct_encoder, img)
    sA = m["stageA"]
    Xs = sA["scaler"].transform(whole_emb[None, :])
    TH_pred = sA["pls"].predict(Xs)[0]
    TH_w = sA["mean_th"] + sA["wgt"] * (TH_pred - sA["mean_th"])
    ilm, rpe = _build_ilm_rpe(TH_w, sA)

    # Stage A/B 학습 데이터(GAMMA 손라벨)가 OD/OS 구분 없이 좌우 flip을 안 한
    # 채로 섞여 학습됐다(위 함수 docstring 참고). OD 기준으로 생성하는 게
    # 기본이므로, 입력이 OS면 두께 profile을 좌우 반전해 nasal/temporal 방향을
    # 실제 눈과 맞춘 뒤 diffusion 조건으로 넘긴다.
    if eye_side == "OS":
        ilm, rpe = ilm[::-1].copy(), rpe[::-1].copy()

    cond = D.build_cond_for(ilm, rpe, DIFF_SPEC)
    cond_t = torch.from_numpy(cond)[None, None].to(DEVICE)

    # DDIM(eta=0)은 결정론적이지만 시작 노이즈가 매번 랜덤이라 speckle 패턴이
    # 샘플마다 달라진다(CLAUDE.md 12절). 5장 생성해 가장 매끄러운(speckle 적은)
    # 1장을 자동 선택 - "더 정확한" 선택이 아니라 "더 보기 좋은" 선택이다(구조는
    # 5장 다 동일 조건이라 같음, speckle만 다름).
    N_SAMPLES = 5

    def _diff_progress(sample_i):
        def _cb(k, total):
            pct = 60 + int(35 * (sample_i + k / total) / N_SAMPLES)
            progress_bar.progress(min(pct, 95), text="합성 OCT 생성 중…")
        return _cb

    candidates = []
    for i in range(N_SAMPLES):
        gen = D.ddim_sample(m["unet"], cond_t, DIFF_SPEC, steps=100,
                            progress_cb=_diff_progress(i))
        img_i = ((gen.clamp(-1, 1) + 1) * 127.5).cpu().numpy()[0, 0].astype(np.uint8)
        candidates.append(img_i)

    def _speckle_score(im):
        """고주파(speckle) 에너지 - 낮을수록 매끄러움(=보기 좋음)."""
        from scipy.ndimage import gaussian_filter
        f = im.astype(np.float32)
        hf = f - gaussian_filter(f, sigma=1.5)
        return float(hf.std())

    gen_img = min(candidates, key=_speckle_score)

    progress_bar.progress(100, text="분석 완료")
    time.sleep(0.2)
    progress_bar.empty()

    concept_dict = dict(zip(ALL_CONCEPTS, concept_vec_raw.tolist()))
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
    """concept 기여도(gradient*input)를 절댓값 크기순 가로 막대로 그린다.
    빨강=위험 방향(+), 파랑=보호 방향(-). Windows 기본 한글 폰트(맑은 고딕)를
    지정해 CONCEPT_META의 한글 라벨을 그대로 표시(영문 키보다 직관적)."""
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
    """cbm.EyeonCBM의 seg_model로 disc/cup을 분할해 원본 fundus 위에 오버레이한
    이미지를 만든다 (사용자 요청: 조건 sketch 대신 실제 C/D segmentation 결과를
    보여줌). retfound_seg.cdr.postprocess_label 규약: 0=배경, disc_rim 값=disc,
    cup 값=cup (glaucoma_cls.eyeon_cbm.EyeonCBM.predict_from_path과 동일 전처리/후처리).

    반환값에 disc 중심의 x좌표(0~1, 이미지 폭 대비 상대위치)도 함께 준다 -
    좌우안(OD/OS) 자동 판별용(_infer_eye_side 참고)."""
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
    # disc 윤곽은 초록, cup 윤곽은 빨강 (반투명 채우기 + 테두리)
    disc_fill = np.zeros_like(base); disc_fill[disc] = [46, 204, 113]
    cup_fill = np.zeros_like(base); cup_fill[cup] = [217, 54, 62]
    overlay = np.where(disc[..., None], overlay * 0.55 + disc_fill * 0.45, overlay)
    overlay = np.where(cup[..., None], overlay * 0.55 + cup_fill * 0.45, overlay)

    disc_cols = np.where(disc.any(axis=0))[0]
    disc_x_rel = float(disc_cols.mean() / disc.shape[1]) if len(disc_cols) else float("nan")

    return overlay.astype(np.uint8), disc_x_rel


def _infer_eye_side(disc_x_rel: float) -> str | None:
    """disc(시신경유두) 중심이 이미지 정중앙보다 오른쪽이면 OD, 왼쪽이면 OS로
    판별한다. laterality.csv(GAMMA 200장) 검증 결과 이미지 중심 대비 disc
    위치만으로 94%(188/200) 정확도, fovea 대비 위치로는 100%(단 fovea 검출기가
    없어 fovea 기준은 못 씀) - 표준 fundus 촬영에서 fovea가 대체로 이미지
    중심 근처에 오기 때문에 '이미지 중심'이 fovea의 실용적 근사치로 작동한다.
    disc가 검출 안 됐거나(OOD 등) 중심에 걸쳐 있으면 판별 보류(None)."""
    if disc_x_rel != disc_x_rel:  # NaN
        return None
    if abs(disc_x_rel - 0.5) < 0.03:  # 중심 근처는 판별 신뢰도 낮음
        return None
    return "OD" if disc_x_rel > 0.5 else "OS"


def _confidence_from_ci(ci: dict) -> tuple:
    """MC-dropout CI 폭으로 신뢰도 등급을 매긴다.
    구간이 좁을수록(예측이 안 흔들릴수록) 확신 높음."""
    width = ci["hi"] - ci["lo"]
    if width < 0.10:
        return "높음", "🟢"
    if width < 0.25:
        return "보통", "🟡"
    return "낮음", "🔴"


# 판정 threshold (raw risk 기준). 재학습마다 재산출해야 하는 값이라 한 곳에만
# 두고 화면/PDF가 모두 여기를 참조한다 - 예전엔 report_pdf.py가 옛 값(21%)을
# 따로 하드코딩하고 있어 리포트에만 폐기된 기준이 찍히는 문제가 있었다.
THR_SUSPECT = 0.60  # "주의 필요" 진입점 (sens=1.000 유지 마지막 지점)
THR_HIGH = 0.66     # "높은 위험" 진입점 (Youden's J 최댓값)


def _risk_grade(risk_prob):
    """위험도 → (등급명, hero CSS 클래스, 한줄결론, 상세설명, 권장행동).

    경계값(0.60 / 0.66)은 REFUGE val 399장(학습에 전혀 안 쓰인 진짜 held-out)
    실측으로 재보정한 값이다(2026-07-30, 24절 crop 수정 후 재학습 반영).
    이전 threshold(0.21/0.5, 2026-07-27 기록)는 letterbox→crop 전처리 수정
    (24절)으로 재학습하며 모델 확률 스케일 자체가 이동해 그대로 재사용하면
    안 되게 됐다. 첫 재산출 시 concept 정규화 통계를 `build_frames()`
    기본값(REFUGE+ORIGA+G1020)으로 잘못 계산해 thr=0.50/0.56이라는 오염된
    값을 냈다가(2026-07-30), train.py가 실제로 쓰는 정규화가
    `build_frames(use_datasets=("REFUGE",))`(REFUGE+GAMMA_train만) 라는 걸
    재확인하고 재계산 - 두 통계가 concept 스케일 차이 때문에 확률
    median을 0.45→0.56로 바꿀 만큼 민감했다(사용자가 REFUGE train
    위음성 이상치를 지적하며 발견, 2026-07-30). 올바른 통계로 thr을
    0.30~0.84(0.02 간격) 재스캔: sens=1.000(FN=0)이 유지되는 마지막 지점이
    0.60(spec=0.710) - 기존과 동일하게 "선별 목적상 위음성 최소화 우선"
    원칙으로 하한값 채택. Youden's J(sens+spec-1) 전체 최댓값은 0.66
    (sens=0.900/spec=0.866, J=0.766)으로 상한값 채택. 재학습마다 이 두 값도
    함께 재산출해야 하며, 그때마다 정규화 통계가 train.py의 build_frames
    호출 인자와 정확히 일치하는지 반드시 재확인할 것(1절 주의사항 참고)."""
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
    """등급(cls: hero-green/amber/red) 기준 고정 위치로 채운다. raw risk %를
    그대로 막대 길이에 쓰면 캘리브레이션 안 된 확률(정상군 대부분 3~10%,
    녹내장군 5~86%로 뭉쳐있음)이 그대로 드러나 오해를 준다 - 환자용 화면은
    판정된 등급만 시각화(2026-07-27, 사용자 요청)."""
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
    """환자용: 결과→의미→근거(쉬운말)→다음 행동 스토리."""
    risk = result["risk_prob"]; ci = result["risk_ci"]; concept = result["concept"]
    cdr = concept.get("cdr", float("nan"))
    grade, cls, headline, detail, action = _risk_grade(risk)
    color = {"hero-green": "#2e9e6b", "hero-amber": "#e08a2b", "hero-red": "#d6483f"}[cls]

    # 히어로 카드 (raw risk %는 캘리브레이션 안 돼 오해 소지가 있어 등급만 표시)
    st.markdown(
        f'<div class="hero-card {cls}">'
        f'<div class="hero-label">AI 녹내장 위험도 선별 결과</div>'
        f'<div class="hero-title">{grade}</div>'
        f'<div class="hero-sub">{headline}</div></div>',
        unsafe_allow_html=True)

    # 위험도 게이지(등급 기준) + 신뢰도 한 줄
    st.markdown(_gauge_html(cls, color), unsafe_allow_html=True)
    conf_txt, conf_icon = _confidence_from_ci(ci)
    st.caption(f"AI 예측 신뢰도 {conf_icon} {conf_txt} "
               f"(비슷한 사진을 반복 분석했을 때 결과가 {'거의 일정' if conf_txt=='높음' else '다소 변동'}함)")

    # 이게 무슨 의미인가요
    _step("이 결과는 무슨 의미인가요?", detail)

    # 왜 이런 결과가 나왔나요 (쉬운말 근거)
    sal = result["saliency"]
    top = max(sal, key=lambda k: abs(sal[k]))
    top_label = CONCEPT_META.get(top, (top,))[0]
    cdr_txt = (f"시신경의 함몰비(C/D 비율)가 {cdr:.2f}로 "
               f"{'다소 높은 편입니다. 이 값이 클수록 녹내장 가능성이 올라갑니다.' if cdr>=0.6 else '정상 범위입니다.'}")
    _step("왜 이런 결과가 나왔나요?",
          f"AI는 시신경유두(눈 안쪽에서 신경이 모이는 부위)의 모양과 망막 구조를 종합해 "
          f"판단했습니다. 이번 분석에서 가장 크게 영향을 준 요소는 <b>{top_label}</b>였고, {cdr_txt} "
          f"오른쪽 ‘상세 분석’ 탭에서 AI가 주목한 부위를 이미지로 확인할 수 있습니다.")

    # 이제 어떻게 해야 하나요
    _step("이제 어떻게 해야 하나요?", action)

    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


_PLATT_A, _PLATT_B = 1.844, 0.554  # REFUGE val(n=399, 진짜 held-out) 재적합,
                                   # 2026-07-27 pos_weight_scale=0.6 재학습 이후.
                                   # LogisticRegression C=0.1(강한 정규화)로
                                   # 재계산 - C 큰 값(약한 정규화)은 held-out
                                   # 표본(양성 40개뿐)에 과적합돼 정상군 max가
                                   # 75%까지 튀는 문제가 있었음(C=10: a=4.76).


def _platt_scale(p):
    """표시 전용 확률 재보정(Platt scaling). raw sigmoid는 정상/녹내장군 확률이
    둘 다 좁은 대역에 몰려있어 체감상 다 낮아 보이는 문제가 있었다. temperature
    scaling(T 하나)은 거의 효과 없어(모델이 과신이 아니라 애초에 확신이 낮은
    상태라 스칼라로는 안 벌어짐), a·logit+b 2파라미터 Platt scaling으로 분포를
    넓힘. **판정(threshold 0.3/0.5)에는 안 쓰고 의료진용 화면의 표시값에만
    적용** - 실측 검증된 threshold는 raw 기준으로 유지해야 sens/spec이 21절
    수치와 어긋나지 않는다."""
    eps = 1e-7
    pc = min(max(p, eps), 1 - eps)
    z = _PLATT_A * np.log(pc / (1 - pc)) + _PLATT_B
    return 1 / (1 + np.exp(-z))


def _render_clinician_view(result):
    """의료진용: 전문 지표·근거 자료·XAI."""
    risk = result["risk_prob"]; ci = result["risk_ci"]; concept = result["concept"]
    cdr = concept.get("cdr", float("nan"))
    conf_txt, conf_icon = _confidence_from_ci(ci)
    risk_disp = _platt_scale(risk)  # 표시용(재보정), 판정은 raw risk로

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

    # XAI 2열
    st.markdown("###### 판단 근거 (XAI)")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.image(result["gradcam_overlay"],
                 caption="Attention Rollout — 모델 주목 영역(참고용, 신뢰도 낮음)",
                 use_container_width=True, clamp=True)
    with ex2:
        st.pyplot(_saliency_figure(result["saliency"]), use_container_width=True)
        st.caption("Concept saliency (gradient×input): 빨강=위험 기여, 파랑=보호 기여 "
                   "— 공간적 히트맵과 달리 이 수치 기반 근거는 별도 검증 대상 아님")

    # 구조 영상 2열
    st.markdown("###### 합성 OCT · 시신경 구조")
    oc1, oc2 = st.columns(2)
    oc1.image(result["oct_image"], caption="합성 OCT B-scan (512×512)",
              use_container_width=True, clamp=True)
    oc2.image(result["seg_overlay"], caption="Disc(초록)/Cup(빨강) Segmentation",
              use_container_width=True, clamp=True)

    # 개념 9종 표
    st.markdown("###### 시신경/망막 정량 지표 (9종)")
    rows = {"지표": [], "값": [], "정상범위": [], "판정": [], "단위": [], "설명": []}
    for c in ALL_CONCEPTS:
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
                # 정상범위를 벗어났지만 위험 방향의 반대쪽(예: 평균보다 두꺼움)이라
                # 위험 신호는 아님 - 정상으로 표시.
                verdict = "🟢 정상범위"
        else:
            rows["정상범위"].append("—")
            verdict = "—"
        rows["판정"].append(verdict)
        rows["단위"].append(unit if unit else "—")
        rows["설명"].append(desc)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("정상범위는 REFUGE+ORIGA+G1020+GAMMA(n=2127) 중 정상 판정군의 IQR(25~75%ile). "
               "mean_th·ilm_rough·fovea_curv는 fundus에서 예측한 OCT 지표(실측 아님). "
               "면적(px)은 512×512 마스크 픽셀 수로 상대 비교용. "
               "성능: AUROC 0.969 (REFUGE val + GAMMA holdout val, n=418). 모델: glaucoma_cls "
               f"(unf{CLS_UNFREEZE_LAST_N}+concept_proj{CLS_CONCEPT_PROJ_DIM}"
               f"+pos_weight{CLS_POS_WEIGHT_SCALE}). 선별 보조용, 확정 진단 아님.")


def _render_result(result):
    """분석 결과를 전체 폭으로 렌더링. result가 None이면 안내만 표시."""
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

    # PDF 리포트 다운로드 (같은 result에 대해 재생성 방지용 캐시)
    # 환자 식별 정보 연동 전이라 임시로 가상 이름을 표시(TODO: 실제 업로드 폼에 이름 입력 추가).
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

    # 환자용 / 의료진용 탭 분리
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

    # 업로드(좌) + 미리보기(우), 결과는 아래 전체 폭
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
