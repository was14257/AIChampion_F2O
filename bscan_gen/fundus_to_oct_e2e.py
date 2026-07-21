import random
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diffusion_bscan as D
from gen_from_handlabels import ddim_sample

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from utils import gamma_fundus_path

LINES = CFG.paths.oct_labels / "lines"
BSC = CFG.paths.oct_labels / "bscans"
EMB = CFG.paths.oct_features / "emb_whole_final.npz"
FEAT = CFG.paths.oct_features / "features.csv"
GMM = CFG.paths.gamma_grading
OUTE = D.OUT / "e2e"
OUTE.mkdir(parents=True, exist_ok=True)
Wc = CFG.oct_tier1.worig
HORIG = CFG.oct_tier1.horig


def fundus_path(c):
    return gamma_fundus_path(c, GMM)


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
    rpe_shape = (RPE - RPE.mean(1, keepdims=True)).mean(0)
    base = float(np.median(RPE))
    x = np.arange(Wc)
    ec = np.abs(x - Wc / 2) / (Wc / 2)
    wgt = gaussian_filter1d(np.clip((ec - 0.08) / 0.17, 0, 1), 10)

    print(f"Stage A: {len(ids)} case 두께 예측 (5-fold OOF)")
    TH_pred = np.zeros_like(TH)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xk):
        sc = StandardScaler().fit(Xk[tr])
        m = PLSRegression(8).fit(sc.transform(Xk[tr]), TH[tr])
        TH_pred[te] = m.predict(sc.transform(Xk[te]))
    TH_w = mean_th + wgt * (TH_pred - mean_th)
    rpe_g = base + rpe_shape
    ILM_g = rpe_g[None] - TH_w
    RPE_g = np.tile(rpe_g, (len(ids), 1))

    print("Stage B: diffusion 생성")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = D.UNet().to(dev)
    net.load_state_dict(torch.load(D.CKPT, map_location=dev, weights_only=False)["model"])
    net.eval()
    conds = torch.stack([
        torch.from_numpy(D.build_cond(ILM_g[i], RPE_g[i]))[None]
        for i in range(len(ids))
    ])

    gens = []
    for s in range(0, len(ids), 16):
        g = ddim_sample(net, conds[s:s + 16].to(dev), steps=100)
        gens.append(((g.clamp(-1, 1) + 1) * 127.5).cpu().numpy()[:, 0])
        print(f"  {min(s + 16, len(ids))}/{len(ids)}")
    gens = np.concatenate(gens)
    for c, g in zip(ids, gens):
        Image.fromarray(g.astype(np.uint8)).save(OUTE / f"{c}_e2e.png")

    random.seed(2)
    pick = random.sample(range(len(ids)), 10)
    fig, ax = plt.subplots(4, 10, figsize=(22, 9))
    for j, idx in enumerate(pick):
        c = ids[idx]
        fund = np.asarray(Image.open(fundus_path(c)).convert("RGB").resize((256, 256)))
        real = np.asarray(Image.open(BSC / f"{c}.png").convert("L").resize((D.W, D.H)))
        ax[0, j].imshow(fund)
        ax[0, j].axis("off")
        ax[0, j].set_title(c, fontsize=8)
        ax[1, j].imshow(conds[idx, 0], cmap="viridis", aspect="auto")
        ax[1, j].axis("off")
        ax[2, j].imshow(gens[idx], cmap="gray", aspect="auto")
        ax[2, j].axis("off")
        ax[3, j].imshow(real, cmap="gray", aspect="auto")
        ax[3, j].axis("off")
    for r, lab in zip(range(4), ["fundus", "pred sketch", "GEN OCT", "real OCT"]):
        ax[r, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle("END-TO-END  fundus -> predicted sketch -> diffusion -> OCT   (bottom: real)", fontsize=13)
    plt.tight_layout()
    plt.savefig(D.OUT / "e2e_compare.png", dpi=95)
    plt.close()
    print(f"비교 → {D.OUT / 'e2e_compare.png'}  |  전체 저장 → {OUTE}")


if __name__ == "__main__":
    main()
