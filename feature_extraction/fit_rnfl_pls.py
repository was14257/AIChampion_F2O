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
    """Load GRAPE baseline RNFL ground truth (mean/S/N/I/T) and matching CFP paths."""
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
    """Estimate the disc's x-center from a segmentation mask, falling back to image center."""
    disc = label_map >= 1
    if disc.sum() == 0:
        return int(img_w * fallback_ratio)
    cols = np.where(disc.any(axis=0))[0]
    seg_w = label_map.shape[1]
    return int((cols.min() + cols.max()) / 2 / seg_w * img_w)


def preprocess_for_seg(img, device):
    """Resize and normalize an image into a tensor batch for the segmentation model."""
    size = CFG.data.img_size
    arr = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CFG.data.mean).view(3, 1, 1)
    std = torch.tensor(CFG.data.std).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)


def extract_embeddings(paths, seg_model, oct_enc):
    """Compute whole-image and disc-crop embeddings for each fundus path, using seg-predicted disc location."""
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
    """Train a PLS regression from fundus embeddings to GRAPE RNFL thickness, report 5-fold CV, and save the final model.

    GRAPE (244 images, measured OCT RNFL) replaces the smaller GAMMA hand-labeled
    fit (fit_oct_linear.py) and covers more severe glaucoma cases. Note: GRAPE is
    all-glaucoma with a lower RNFL range than REFUGE/GAMMA, so there is a domain
    shift - treat predictions as relative (correlation) rather than absolute values.
    """
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

    # Final fit on all data, saved for REFUGE/GAMMA pseudo-label generation
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
