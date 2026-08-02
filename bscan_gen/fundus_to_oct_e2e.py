"""End-to-end fundus -> predicted sketch -> OCT generation (512 resolution),
applying the same "generate 5, pick the one with least speckle" logic used
by the web demo (app_streamlit.py).

Rationale (app_streamlit.py, around line 458): DDIM (eta=0) is deterministic,
but the starting noise is random each time, so the speckle pattern differs
between samples. Generate 5 and auto-pick the smoothest (least speckle) one.
This is not a "more accurate" pick, just a "nicer-looking" one - since the
condition is identical, the structure is the same across all 5, only the
speckle noise differs.

Output: outputs/diffusion/e2e_512_best5/{case_id}_e2e.png (all 172)
     + outputs/diffusion/e2e_512_best5_compare.png (10-sample comparison grid)
"""
import random
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bscan_gen import diffusion_bscan as D
from bscan_gen.utils import gamma_fundus_path
from config import CFG

# 512 model, stated explicitly instead of patching D.H/D.W/D.CKPT at import
# time - that used to leak into every other module that imported D.
SPEC = D.spec_512()
assert SPEC.ckpt.exists(), f"checkpoint not found: {SPEC.ckpt}"

LINES = CFG.paths.oct_labels / "lines"
BSC = CFG.paths.oct_labels / "bscans"
EMB = CFG.paths.oct_features / "emb_whole_final.npz"
FEAT = CFG.paths.oct_features / "features.csv"
GMM = CFG.paths.gamma_grading
OUTE = D.OUT / "e2e_512_best5"
OUTE.mkdir(parents=True, exist_ok=True)
Wc = CFG.oct_tier1.worig
HORIG = CFG.oct_tier1.horig

N_SAMPLES = 5  # same as app_streamlit.py


def fundus_path(c):
    return gamma_fundus_path(c, GMM)


def speckle_score(im):
    """Same as app_streamlit.py::_speckle_score - high-frequency energy, lower is smoother."""
    f = im.astype(np.float32)
    hf = f - gaussian_filter(f, sigma=1.5)
    return float(hf.std())


def main():
    df = pd.read_csv(FEAT, dtype={"case_id": str})
    df["case_id"] = df["case_id"].str.zfill(4)
    z = np.load(EMB, allow_pickle=True)
    assert list(z["ids"]) == list(df["case_id"])
    X = z["X"]
    ids = df["case_id"].values

    ILM, RPE, keep = [], [], []
    for i, c in enumerate(ids):
        p = LINES / f"{c}.npz"
        if p.exists():
            d = np.load(p)
            if not (np.isnan(d["ilm"]).any() or np.isnan(d["rpe"]).any()):
                ILM.append(d["ilm"])
                RPE.append(d["rpe"])
                keep.append(i)
    ILM = np.stack(ILM)
    RPE = np.stack(RPE)
    Xk = X[keep]
    ids = ids[keep]

    TH = RPE - ILM
    mean_th = TH.mean(0)
    rpe_shapes = RPE - RPE.mean(1, keepdims=True)  # per-case RPE shape (deviation from mean)
    base = float(np.median(RPE))
    x = np.arange(Wc)
    ec = np.abs(x - Wc / 2) / (Wc / 2)
    wgt = gaussian_filter1d(np.clip((ec - 0.08) / 0.17, 0, 1), 10)

    print(f"Stage A: predicting thickness for {len(ids)} cases (5-fold OOF)")
    TH_pred = np.zeros_like(TH)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xk):
        sc = StandardScaler().fit(Xk[tr])
        m = PLSRegression(8).fit(sc.transform(Xk[tr]), TH[tr])
        TH_pred[te] = m.predict(sc.transform(Xk[te]))
    TH_w = mean_th + wgt * (TH_pred - mean_th)

    nn_idx = np.argmin(
        ((TH_w[:, None, :] - TH[None, :, :]) ** 2).mean(-1), axis=1
    )
    rpe_g = base + rpe_shapes[nn_idx]
    ILM_g = rpe_g - TH_w
    RPE_g = rpe_g

    print(f"Stage B: diffusion generation (generate {N_SAMPLES} per case, pick the one with least speckle, same logic as app_streamlit.py)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = D.load_unet(SPEC, dev)
    conds = torch.stack([
        torch.from_numpy(D.build_cond_for(ILM_g[i], RPE_g[i], SPEC))[None]
        for i in range(len(ids))
    ])

    gens = []
    for i in range(len(ids)):
        cond_i = conds[i:i + 1].to(dev)
        candidates = []
        for _ in range(N_SAMPLES):
            g = D.ddim_sample(net, cond_i, SPEC, steps=100)
            im = ((g.clamp(-1, 1) + 1) * 127.5).cpu().numpy()[0, 0].astype(np.uint8)
            candidates.append(im)
        best = min(candidates, key=speckle_score)
        gens.append(best)
        if (i + 1) % 8 == 0 or i == len(ids) - 1:
            print(f"  {i + 1}/{len(ids)}")
    gens = np.stack(gens)

    for c, g in zip(ids, gens):
        Image.fromarray(g).save(OUTE / f"{c}_e2e.png")

    random.seed(2)
    pick = random.sample(range(len(ids)), 10)
    fig, ax = plt.subplots(4, 10, figsize=(22, 9))
    for j, idx in enumerate(pick):
        c = ids[idx]
        fund = np.asarray(Image.open(fundus_path(c)).convert("RGB").resize((256, 256)))
        real = np.asarray(Image.open(BSC / f"{c}.png").convert("L").resize((SPEC.w, SPEC.h)))
        ax[0, j].imshow(fund)
        ax[0, j].axis("off")
        ax[0, j].set_title(c, fontsize=8)
        ax[1, j].imshow(conds[idx, 0], cmap="viridis", aspect="auto")
        ax[1, j].axis("off")
        ax[2, j].imshow(gens[idx], cmap="gray", aspect="auto")
        ax[2, j].axis("off")
        ax[3, j].imshow(real, cmap="gray", aspect="auto")
        ax[3, j].axis("off")
    for r, lab in zip(range(4), ["fundus", "pred sketch", "GEN OCT (best-of-5)", "real OCT"]):
        ax[r, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle("END-TO-END (best-of-5, speckle-selected)  fundus -> predicted sketch -> diffusion -> OCT   (bottom: real)", fontsize=13)
    plt.tight_layout()
    out_compare = D.OUT / "e2e_512_best5_compare.png"
    plt.savefig(out_compare, dpi=95)
    plt.close()
    print(f"comparison -> {out_compare}  |  full output -> {OUTE}")


if __name__ == "__main__":
    main()
