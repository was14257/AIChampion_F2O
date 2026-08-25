"""GRAPE(244장, 실측 OCT RNFL Mean/S/N/I/T 보유)로 fundus embedding -> RNFL
thickness 회귀를 학습한다. GAMMA 172장 hand-labeled 학습(fit_oct_linear.py)의
대체재: GRAPE는 표본이 더 많고(244 vs 172) 전원 녹내장이라 더 넓은 중증도
범위(RNFL 저하가 심한 쪽)를 커버한다.

seg_model로 disc 위치를 잡아 disc crop을 만들고, whole+disc embedding(224
encoder)을 concat해 PLS 회귀를 적용한다 (fit_oct_linear.py와 동일 방식).

5-fold CV 결과(2026-08-09): mean_th r=0.828, S r=0.763, I r=0.766, T r=0.725
모두 baseline(평균만 예측)보다 뚜렷이 우수. N(비강측)만 r=0.393으로 약함 -
N은 사분면 중 개인차(std)가 가장 작고 다른 사분면과의 상관도 가장 낮아
(정상군에서도 임상적으로 가장 늦게 손상되는 부위), fundus로 예측하기 어려운
게 자연스럽다.

GRAPE 내부 CV에서는 좋은 성능을 보이지만, GRAPE(전원 중증 녹내장, RNFL
44~119 낮은 범위)와 REFUGE/GAMMA(정상~경증 포함, 더 넓은 분포) 간 도메인
시프트가 있어 절대값은 GRAPE 분포 쪽으로 편향될 수 있다 - 순위(상관) 정보로
활용하는 것이 안전하다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from bscan_gen.utils import load_retfound_encoder, embed_fundus, disc_crop
from retfound_seg.model import build_model as build_seg_model
from retfound_seg.cdr import postprocess_label

GRAPE_ROOT = CFG.paths.grape
CFP_DIR = GRAPE_ROOT / "CFPs"
XLSX = GRAPE_ROOT / "VF and clinical information.xlsx"
DEVICE = CFG.runtime.device
TARGETS = ["mean_th", "S", "N", "I", "T"]
OUT_PKL = CFG.paths.oct_features / "grape_rnfl_pls.pkl"


def load_grape_gt():
    df = pd.read_excel(XLSX, sheet_name="Baseline", header=None, skiprows=2)
    out = df[[16, 11, 12, 13, 14, 15]].copy()
    out.columns = ["filename"] + TARGETS
    for c in TARGETS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["filename"] + TARGETS)
    out["path"] = out["filename"].apply(lambda f: str(CFP_DIR / str(f).strip()))
    out = out[out["path"].apply(lambda p: Path(p).exists())]
    return out.reset_index(drop=True)


def disc_x_from_mask(label_map, img_w, fallback_ratio=0.5):
    disc = label_map >= 1
    if disc.sum() == 0:
        return int(img_w * fallback_ratio)
    cols = np.where(disc.any(axis=0))[0]
    seg_w = label_map.shape[1]
    return int((cols.min() + cols.max()) / 2 / seg_w * img_w)


def preprocess_for_seg(img, device):
    size = CFG.data.img_size
    arr = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CFG.data.mean).view(3, 1, 1)
    std = torch.tensor(CFG.data.std).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)


def extract_embeddings(paths, seg_model, oct_enc):
    Wh, Di = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            logits = seg_model(preprocess_for_seg(img, DEVICE))
        label_map = postprocess_label(logits.argmax(dim=1)[0].cpu().numpy())
        dx = disc_x_from_mask(label_map, img.width)
        cropped = disc_crop(img, dx)
        Wh.append(embed_fundus(oct_enc, img))
        Di.append(embed_fundus(oct_enc, cropped))
    return np.stack(Wh), np.stack(Di)


def main():
    gt = load_grape_gt()
    print(f"GRAPE 실측 RNFL 보유: {len(gt)}장")

    seg = build_seg_model(load_weights=True)
    seg_ck = torch.load(CFG.paths.ckpt_dir / "best.pth", map_location="cpu", weights_only=False)
    seg.load_state_dict(seg_ck["model"])
    seg.eval().to(DEVICE)

    oct_enc = load_retfound_encoder()
    oct_enc.eval().to(DEVICE)

    print("임베딩 추출 중...")
    Wh, Di = extract_embeddings(gt["path"].tolist(), seg, oct_enc)
    X = np.hstack([Wh, Di])
    Y = gt[TARGETS].to_numpy(dtype=np.float64)
    print(f"X={X.shape}  Y={Y.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_mse = {t: [] for t in TARGETS}
    fold_r = {t: [] for t in TARGETS}
    baseline_mse = {t: [] for t in TARGETS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        Xtr, Xva = X[tr_idx], X[va_idx]
        Ytr, Yva = Y[tr_idx], Y[va_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

        pls = PLSRegression(n_components=8).fit(Xtr_s, Ytr)
        pred = pls.predict(Xva_s)

        for i, t in enumerate(TARGETS):
            mse = np.mean((pred[:, i] - Yva[:, i]) ** 2)
            r = np.corrcoef(pred[:, i], Yva[:, i])[0, 1]
            base = np.mean((Ytr[:, i].mean() - Yva[:, i]) ** 2)
            fold_mse[t].append(mse)
            fold_r[t].append(r)
            baseline_mse[t].append(base)
        print(f"fold {fold}: done (n_tr={len(tr_idx)}, n_va={len(va_idx)})")

    print("\n=== 5-fold CV 결과 (GRAPE 내부) ===")
    for t in TARGETS:
        mse_m, mse_s = np.mean(fold_mse[t]), np.std(fold_mse[t])
        r_m, r_s = np.mean(fold_r[t]), np.std(fold_r[t])
        base_m = np.mean(baseline_mse[t])
        print(f"{t:<10} RMSE={np.sqrt(mse_m):6.2f}±{np.sqrt(mse_s):.2f}  "
              f"r={r_m:.3f}±{r_s:.3f}  baseline_RMSE={np.sqrt(base_m):.2f}")

    # 최종적으로 244장 전체로 fit -> 저장 (REFUGE/GAMMA pseudo-label 생성용)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pls_final = PLSRegression(n_components=8).fit(Xs, Y)

    import pickle
    with open(OUT_PKL, "wb") as f:
        pickle.dump({"scaler": scaler, "pls": pls_final, "targets": TARGETS,
                     "in_dim": X.shape[1]}, f)
    print(f"\n최종 모델 저장: {OUT_PKL}")


if __name__ == "__main__":
    main()
