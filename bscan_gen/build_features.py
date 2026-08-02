import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pointbiserialr, mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG

LINES = CFG.paths.oct_labels / "lines"
CHOSEN = CFG.paths.oct_labels / "chosen_slices.csv"
LAT = CFG.paths.oct_labels / "laterality.csv"

GMM = CFG.paths.gamma_grading
GT_XLSX = GMM / "training/glaucoma_grading_training_GT.xlsx"

OUT = CFG.paths.oct_features
OUT.mkdir(parents=True, exist_ok=True)


def _band(th, x0, x1):
    W = len(th)
    a, b = int(x0 * W), int(x1 * W)
    seg = th[a:max(a + 1, b)]
    return float(np.nanmean(seg))


def case_features(ilm, rpe, laterality):
    th = rpe - ilm
    W = len(th)
    c = W // 2
    half = max(3, int(0.03 * W))

    cft = float(np.nanmean(th[c - half:c + half + 1]))
    lo, hi = int(0.35 * W), int(0.65 * W)
    fovea_min = float(np.nanmin(th[lo:hi]))
    fmin_pos = (int(np.nanargmin(th[lo:hi])) + lo) / W - 0.5
    para = np.concatenate([th[:int(0.2 * W)], th[int(0.8 * W):]])
    para_mean = float(np.nanmean(para))
    pit_depth = para_mean - fovea_min
    mean_th = float(np.nanmean(th))
    std_th = float(np.nanstd(th))
    max_th = float(np.nanmax(th))

    inner_L = _band(th, 0.35, 0.45)
    inner_R = _band(th, 0.55, 0.65)
    outer_L = _band(th, 0.05, 0.25)
    outer_R = _band(th, 0.75, 0.95)
    inner_ring = 0.5 * (inner_L + inner_R)
    outer_ring = 0.5 * (outer_L + outer_R)

    left = float(np.nanmean(th[:c]))
    right = float(np.nanmean(th[c:]))
    if laterality == "OD":
        nasal, temporal = right, left
        nasal_inner, temporal_inner = inner_R, inner_L
        nasal_outer, temporal_outer = outer_R, outer_L
    else:
        nasal, temporal = left, right
        nasal_inner, temporal_inner = inner_L, inner_R
        nasal_outer, temporal_outer = outer_L, outer_R

    nt_ratio = nasal / max(1e-6, temporal)
    nt_diff = nasal - temporal
    inner_asym = nasal_inner - temporal_inner
    outer_asym = nasal_outer - temporal_outer

    pit_slope = pit_depth / max(1e-6, 0.15 * W)
    ilm_detr = ilm - gaussian_filter1d(ilm, 30)
    ilm_rough = float(np.nanstd(ilm_detr))

    ilm_s = gaussian_filter1d(ilm, 8)
    cw = int(0.08 * W)
    d2 = np.gradient(np.gradient(ilm_s[c - cw:c + cw + 1]))
    fovea_curv = float(np.mean(d2))
    peak_curv = float(np.max(np.abs(d2)))
    rpe_curv = float(np.nanstd(rpe - gaussian_filter1d(rpe, 60)))
    thick_grad = float(np.nanmean(np.abs(np.diff(th))))

    return dict(
        cft=cft, fovea_min=fovea_min, fmin_pos=fmin_pos, para_mean=para_mean,
        pit_depth=pit_depth, pit_slope=pit_slope, mean_th=mean_th, std_th=std_th,
        max_th=max_th, inner_ring=inner_ring, outer_ring=outer_ring,
        nasal_mean=nasal, temporal_mean=temporal, nt_ratio=nt_ratio, nt_diff=nt_diff,
        nasal_inner=nasal_inner, temporal_inner=temporal_inner,
        inner_asym=inner_asym, outer_asym=outer_asym,
        ilm_rough=ilm_rough, rpe_curv=rpe_curv, thick_grad=thick_grad,
        fovea_curv=fovea_curv, peak_curv=peak_curv)


def load_labels():
    lat = pd.read_csv(LAT, encoding="utf-8-sig")
    lat["case_id"] = lat["case_id"].astype(str).str.zfill(4)
    lat = lat.set_index("case_id")["laterality"].to_dict()

    gt = pd.read_excel(GT_XLSX)
    lab = {}
    for _, r in gt.iterrows():
        lab[f"{int(r['data']):04d}"] = 0 if int(r["non"]) == 1 else 1
    return lat, lab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lat, lab = load_labels()
    rows = []
    for npz in sorted(LINES.glob("*.npz")):
        cid = npz.stem
        d = np.load(npz)
        if np.isnan(d["ilm"]).all() or np.isnan(d["rpe"]).all():
            print(f"  {cid}: empty label, skip")
            continue
        f = case_features(d["ilm"], d["rpe"], lat.get(cid, "OD"))
        f["case_id"] = cid
        f["laterality"] = lat.get(cid, "?")
        f["glaucoma"] = lab.get(cid, np.nan)
        rows.append(f)

    df = pd.DataFrame(rows).set_index("case_id")
    df.to_csv(OUT / "features.csv", encoding="utf-8-sig")
    print(f"\n{len(df)} case features extracted -> {OUT/'features.csv'}")
    print("Glaucoma distribution:", df["glaucoma"].value_counts().to_dict())

    feats = [c for c in df.columns if c not in ("laterality", "glaucoma")]
    sub = df.dropna(subset=["glaucoma"])
    y = sub["glaucoma"].astype(int).values
    lines_out = [f"n={len(sub)}  glaucoma={int(y.sum())} normal={int((y==0).sum())}",
                 f"{'feature':<14}{'normal':>8}{'glaucoma':>9}{'AUC':>7}{'r':>7}{'p':>9}"]
    scored = []
    for ft in feats:
        v = sub[ft].values.astype(float)
        m0, m1 = v[y == 0].mean(), v[y == 1].mean()
        try:
            auc = roc_auc_score(y, v)
            auc = max(auc, 1 - auc)
            r, _ = pointbiserialr(y, v)
            _, p = mannwhitneyu(v[y == 1], v[y == 0], alternative="two-sided")
        except ValueError:
            auc = r = p = float("nan")
        scored.append((ft, m0, m1, auc, r, p))
    for ft, m0, m1, auc, r, p in sorted(scored, key=lambda t: -t[3]):
        star = "*" * sum(p < t for t in (0.05, 0.01, 0.001)) if p == p else ""
        lines_out.append(f"{ft:<14}{m0:>8.1f}{m1:>9.1f}{auc:>7.3f}{r:>7.2f}{p:>9.4f} {star}")
    lines_out.append("\n* p<.05  ** p<.01  *** p<.001 (Mann-Whitney)")
    report = "\n".join(lines_out)
    (OUT / "feature_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    top = [t[0] for t in sorted(scored, key=lambda t: -t[3])[:4]]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, ft in zip(axes, top):
        ax.boxplot([sub[sub.glaucoma == 0][ft], sub[sub.glaucoma == 1][ft]],
                   labels=["normal", "glaucoma"])
        ax.set_title(ft)
    plt.tight_layout()
    plt.savefig(OUT / "feature_box.png", dpi=100)
    print(f"\nBoxplot -> {OUT/'feature_box.png'}")


if __name__ == "__main__":
    main()
