import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from utils import gamma_fundus_path, load_retfound_encoder, embed_fundus, disc_crop

OUT = CFG.paths.oct_features
PSE = CFG.paths.oct_pseudo / "lines"
GMM = CFG.paths.gamma_grading
LINES = CFG.paths.oct_labels / "lines"
LAT = CFG.paths.oct_labels / "laterality.csv"
CACHE = OUT / "emb_dual_all.npz"
Wc = CFG.oct_tier1.worig
HORIG = CFG.oct_tier1.horig


def fundus_path(c):
    return gamma_fundus_path(c, GMM)


def thickness(npz):
    d = np.load(npz)
    return d["rpe"] - d["ilm"]


def main():
    lat = pd.read_csv(LAT, encoding="utf-8-sig")
    lat["case_id"] = lat["case_id"].astype(str).str.zfill(4)
    disc = lat.set_index("case_id")["disc_x"].to_dict()

    df = pd.read_csv(OUT / "features.csv", dtype={"case_id": str})
    df["case_id"] = df["case_id"].str.zfill(4)
    hand = [
        c for c in df["case_id"]
        if (LINES / f"{c}.npz").exists()
        and not (np.isnan(np.load(LINES / f"{c}.npz")["ilm"]).any()
                 or np.isnan(np.load(LINES / f"{c}.npz")["rpe"]).any())
    ]
    TH_hand = np.stack([thickness(LINES / f"{c}.npz") for c in hand])

    vols = {}
    for f in PSE.glob("*.npz"):
        cid, si = f.stem.split("_")
        vols.setdefault(cid, []).append(int(si))
    extra = [c for c in vols if c not in set(hand)]

    ex_th, ex_ids = [], []
    for c in extra:
        si = min(vols[c], key=lambda s: abs(s - 127))
        ex_th.append(thickness(PSE / f"{c}_{si}.npz"))
        ex_ids.append(c)
    TH_ex = np.stack(ex_th)

    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        Wh, Di, Wh_e, Di_e = z["Wh"], z["Di"], z["Wh_e"], z["Di_e"]
        print("임베딩 캐시 사용")
    else:
        enc = load_retfound_encoder()
        print(f"임베딩 계산: 손라벨{len(hand)}+추가{len(ex_ids)} × (whole+disc)")

        def dual(c):
            im = Image.open(fundus_path(c)).convert("RGB")
            dx = disc.get(c, np.asarray(im).shape[1] // 2)
            return embed_fundus(enc, im), embed_fundus(enc, disc_crop(im, dx))

        Wh, Di = zip(*[dual(c) for c in hand])
        Wh, Di = np.stack(Wh), np.stack(Di)
        Wh_e, Di_e = zip(*[dual(c) for c in ex_ids])
        Wh_e, Di_e = np.stack(Wh_e), np.stack(Di_e)
        np.savez(CACHE, Wh=Wh, Di=Di, Wh_e=Wh_e, Di_e=Di_e, hand=hand, ex=ex_ids)

    mean_th = TH_hand.mean(0)
    x = np.arange(Wc)
    ec = np.abs(x - Wc / 2) / (Wc / 2)
    wgt = gaussian_filter1d(np.clip((ec - 0.08) / 0.17, 0, 1), 10)

    def skill(P):
        return 1 - np.sum((TH_hand - P) ** 2) / np.sum((TH_hand - mean_th) ** 2)

    def run(feat_hand, feat_ex=None):
        THp = np.zeros_like(TH_hand)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(feat_hand):
            Xtr, Ytr = feat_hand[tr], TH_hand[tr]
            if feat_ex is not None:
                Xtr = np.vstack([Xtr, feat_ex])
                Ytr = np.vstack([Ytr, TH_ex])
            sc = StandardScaler().fit(Xtr)
            m = PLSRegression(8).fit(sc.transform(Xtr), Ytr)
            THp[te] = m.predict(sc.transform(feat_hand[te]))
        THw = mean_th + wgt * (THp - mean_th)

        rs = {}
        for zn, (a, b) in [("fovea", (0.45, 0.55)), ("para", (0.30, 0.45)), ("outer", (0.05, 0.25))]:
            lo, hi = int(a * Wc), int(b * Wc)
            rs[zn] = pearsonr(TH_hand[:, lo:hi].mean(1), THw[:, lo:hi].mean(1))[0]
        return skill(THw), rs

    print(f"\n{'방법':<26}{'skill':>8}  {'fovea':>6}{'para':>6}{'outer':>6}")
    methods = [
        ("① whole만(기준)", Wh, None),
        ("② whole+disc", np.hstack([Wh, Di]), None),
        ("③ +extra28(whole)", Wh, Wh_e),
        ("④ whole+disc+extra28", np.hstack([Wh, Di]), np.hstack([Wh_e, Di_e])),
    ]
    for nm, fh, fe in methods:
        s, rs = run(fh, fe)
        print(f"{nm:<26}{s:>8.3f}  {rs['fovea']:>6.2f}{rs['para']:>6.2f}{rs['outer']:>6.2f}")


if __name__ == "__main__":
    main()
