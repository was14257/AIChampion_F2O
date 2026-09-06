from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LOCAL = Path("C:/Users/hogri/OneDrive/Desktop/AIGS자율공모/Code/data")
_SERVER = Path("/home/tta/data")
DATA = _LOCAL if _LOCAL.exists() else _SERVER
_GD = Path("D:/GAMMA")
GAMMA = _GD if _GD.exists() else DATA / "GAMMA"
GMM = GAMMA / "grading/Glaucoma_grading"

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _letterbox_square(img: Image.Image) -> Image.Image:
    """Center-crop to a square while preserving the original aspect ratio.

    GAMMA (mostly 2992x2000, ratio~1.5) is mixed with REFUGE/ORIGA/G1020
    (ratio~1.0-1.24); if the subsequent Resize((size,size)) ignored aspect
    ratio and forced a square, images would get squashed by a different
    direction/degree per dataset. Even within GAMMA, a small subgroup shot
    with different equipment (6/100, ratio~1.01) happens to correlate with
    the label (all non-glaucoma), so this distortion pattern risks being
    learned as a shortcut.

    Why crop instead of padding (adding black margins top/bottom): measured
    result (2026-07-30) — for wide GAMMA images (2992x2000), the fundus fills
    the canvas nearly edge-to-edge vertically, and only the horizontal margins
    (~350px on each side) are pure black background (brightness<1). So the
    fundus itself is already near-square and only the extra horizontal camera
    frame needs trimming — padding would introduce a new confound where the
    fundus's scale within the canvas differs by dataset, so a short-side
    center crop is correct."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _refuge():
    """Loads REFUGE train image rows (label from filename prefix 'g')."""
    d = DATA / "REFUGE/train/Images"
    return [(str(p), 1 if p.stem.lower().startswith("g") else 0, "REFUGE")
            for p in sorted(d.glob("*.jpg"))]


def _refuge_val():
    """REFUGE val, 400 images (label determined from index.json's Label field,
    see section 22). By default this is held-out-eval-only and not added to
    build_frames's use_datasets pool; refuge_val_train_frac explicitly folds
    in only a portion for training."""
    import json
    d = json.load(open(DATA / "REFUGE/val/index.json", encoding="utf-8"))
    dr = DATA / "REFUGE/val/Images"
    out = []
    for _, v in d.items():
        p = dr / v["ImgName"]
        if p.exists():
            out.append((str(p), int(v["Label"]), "REFUGE_val"))
    return out


def _origa():
    """Loads ORIGA image rows with labels from OrigaList.csv."""
    df = pd.read_csv(DATA / "ORIGA/OrigaList.csv")
    d = DATA / "ORIGA/Images"
    out = []
    for _, r in df.iterrows():
        p = d / str(r["Filename"])
        if p.exists():
            out.append((str(p), int(r["Glaucoma"]), "ORIGA"))
    return out


def _g1020():
    """Loads G1020 image rows with labels from G1020.csv."""
    df = pd.read_csv(DATA / "G1020/G1020.csv")
    d = DATA / "G1020/Images"
    out = []
    for _, r in df.iterrows():
        p = d / str(r["imageID"])
        if p.exists():
            out.append((str(p), int(r["binaryLabels"]), "G1020"))
    return out


def _gamma_train():
    """Loads GAMMA training image rows with labels from the grading GT excel."""
    df = pd.read_excel(GMM / "training/glaucoma_grading_training_GT.xlsx")
    out = []
    for _, r in df.iterrows():
        cid = f"{int(r['data']):04d}"
        p = GMM / f"training/multi-modality_images/{cid}/{cid}.jpg"
        if p.exists():
            lab = 0 if int(r["non"]) == 1 else 1
            out.append((str(p), lab, "GAMMA_train", cid))
    return out


def _gamma_test():
    """Loads GAMMA test image paths (labels hidden), keyed by case_id."""
    d = GMM / "testing/multi-modality_images"
    out = []
    for cd in sorted(d.iterdir()):
        if cd.is_dir():
            p = cd / f"{cd.name}.jpg"
            if p.exists():
                out.append((str(p), cd.name))
    return out


def refuge_val_split(refuge_val_train_frac, seed=42):
    """Split REFUGE val's 399 images into (training portion, remaining
    held-out portion) using exactly the same stratification rule as
    build_frames(refuge_val_train_frac=...) (per-label, fixed seed). Only the
    training portion is merged into ext_all inside build_frames; the
    remaining held-out portion is not returned (to preserve backward
    compatibility) — callers needing the held-out portion should call this
    function directly to reproduce it (section 25)."""
    rv = pd.DataFrame(_refuge_val(), columns=["path", "label", "dataset"])
    tr_parts, ho_parts = [], []
    for lab, grp in rv.groupby("label"):
        grp = grp.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_tr = int(round(len(grp) * refuge_val_train_frac))
        tr_parts.append(grp.iloc[:n_tr])
        ho_parts.append(grp.iloc[n_tr:])
    train_part = pd.concat(tr_parts, ignore_index=True)
    holdout_part = pd.concat(ho_parts, ignore_index=True).reset_index(drop=True)
    return train_part, holdout_part


def build_frames(gamma_val_frac=0.2, seed=42, include_gamma_train=True,
                use_datasets=("REFUGE", "ORIGA", "G1020"), fold_idx=None, n_folds=5,
                ext_val_frac=0.0, refuge_val_train_frac=0.0):
    """REFUGE/ORIGA/G1020 (ext) and GAMMA_train differ in imaging
    equipment/population/ethnicity (G1020=German/European, ORIGA=Singapore
    Malay, REFUGE=Chinese, GAMMA=Chinese), so training on ext alone showed a
    domain shift that didn't generalize well to GAMMA (val AUC stuck around
    0.65-0.70). We split GAMMA_train's 100 images 80/20, mixing 80 into
    training (ext_all) and keeping 20 as a true holdout (val), bringing
    GAMMA-domain signal into training without leakage.

    include_gamma_train=False: for ablation - drops the 80 GAMMA_train
    images from training entirely, to measure how much performance drops
    when the target ethnicity/domain is never seen.

    use_datasets: choose which external datasets to use for training.
    Dropping G1020 (German/European) and using only ("REFUGE","ORIGA") gives
    an all-Asian (Chinese+Malay) training set that's ethnically closer to
    GAMMA (Chinese) - though the size shrinks from 2070->1050.

    When fold_idx (0..n_folds-1) is given: splits the seed-shuffled 100 GAMMA
    images into n_folds parts and uses only the fold_idx-th part as val,
    performing "true" k-fold (no overlap). If None, behaves as before,
    independently drawing a random 80/20 for each seed - in this case the
    same 20 can be drawn into val multiple times across different seeds
    (not without-replacement), so the mean/variance across 5 repeats can be
    noisier than true 5-fold.

    ext_val_frac>0: validating on GAMMA alone (n=20~100) makes per-fold AUC
    swing too much (user's observation, 2026-07-24). Stratified-sample this
    fraction from each of REFUGE/ORIGA/G1020 into ext_val, and use only the
    remainder for training. ext_val is returned combined with GAMMA val (3rd
    return value) - this reduces variance by increasing val n, but dilutes
    the meaning of "pure GAMMA (target) validation," mixing in
    REFUGE/ORIGA/G1020 (non-target domains).

    refuge_val_train_frac (2026-07-31, section 25): until now, all 399
    REFUGE val images were used purely as "true held-out" for evaluation and
    never fed into training (only n=480 used for training, less than half of
    the full candidate pool n=900). Per the user's request to increase the
    training count, REFUGE val is stratified by class (per-label, fixed
    seed) and only this fraction is added to ext_all (folding in all of
    REFUGE val would remove the baseline for reproducing sections 22/25's
    held-out performance, so we don't fold in the full set). For backward
    compatibility with existing callers, the return tuple stays at 3 items -
    if the remaining REFUGE val excluded from training (new held-out) is
    needed, call `refuge_val_split()` separately with the same seed/frac to
    reproduce exactly the same split."""
    pool = {"REFUGE": _refuge(), "ORIGA": _origa(), "G1020": _g1020()}
    rows = [r for name in use_datasets for r in pool[name]]
    ext = pd.DataFrame(rows, columns=["path", "label", "dataset"])

    ext_val = pd.DataFrame(columns=["path", "label", "dataset"])
    if ext_val_frac > 0:
        val_parts, keep_parts = [], []
        for lab, grp in ext.groupby("label"):
            grp = grp.sample(frac=1, random_state=seed).reset_index(drop=True)
            n_v = max(1, int(len(grp) * ext_val_frac))
            val_parts.append(grp.iloc[:n_v])
            keep_parts.append(grp.iloc[n_v:])
        ext_val = pd.concat(val_parts, ignore_index=True)
        ext = pd.concat(keep_parts, ignore_index=True)

    gamma = pd.DataFrame(_gamma_train(), columns=["path", "label", "dataset", "case_id"])
    gamma = gamma.sample(frac=1, random_state=seed).reset_index(drop=True)
    if fold_idx is not None:
        n = len(gamma)
        bounds = [round(i * n / n_folds) for i in range(n_folds + 1)]
        lo, hi = bounds[fold_idx], bounds[fold_idx + 1]
        val = gamma.iloc[lo:hi].reset_index(drop=True)
        gamma_tr = pd.concat([gamma.iloc[:lo], gamma.iloc[hi:]], ignore_index=True)
    else:
        n_val = max(1, int(len(gamma) * gamma_val_frac))
        val = gamma.iloc[:n_val].reset_index(drop=True)
        gamma_tr = gamma.iloc[n_val:].reset_index(drop=True)
    if len(ext_val) > 0:
        val = pd.concat([val, ext_val[["path", "label", "dataset"]]], ignore_index=True)
    if include_gamma_train:
        ext_all = pd.concat([ext, gamma_tr[["path", "label", "dataset"]]], ignore_index=True)
    else:
        ext_all = ext

    if refuge_val_train_frac > 0:
        rv_train, _ = refuge_val_split(refuge_val_train_frac, seed=seed)
        ext_all = pd.concat([ext_all, rv_train], ignore_index=True)

    test = pd.DataFrame(_gamma_test(), columns=["path", "case_id"])
    return ext_all, val, test


_CBM_CONCEPTS_NPZ = DATA.parent / "outputs/oct_features/cbm_concepts.npz"
_CBM_CONCEPTS_NPZ = (_CBM_CONCEPTS_NPZ if _CBM_CONCEPTS_NPZ.exists()
                     else Path("/home/tta/outputs/oct_features/cbm_concepts.npz"))

_CBM_CONCEPTS_V2_NPZ = DATA.parent / "outputs/oct_features/cbm_concepts_v2.npz"
_CBM_CONCEPTS_V2_NPZ = (_CBM_CONCEPTS_V2_NPZ if _CBM_CONCEPTS_V2_NPZ.exists()
                        else Path("/home/tta/outputs/oct_features/cbm_concepts_v2.npz"))


def load_concept_table():
    """Returns cbm_concepts.npz (9 concept types for REFUGE+G1020+ORIGA+GAMMA)
    as a filename (basename) -> raw concept vector (9,) dict (not
    normalized - normalization stats must be computed only on each fold's
    train set so val info doesn't leak, see attach_concepts). GAMMA has two
    copies (D:/GAMMA and Code/Data/GAMMA) with different path strings (still
    different even after resolve()), so absolute-path matching fails -
    filenames are all unique (case_id.jpg etc.) so basename matching is used
    instead."""
    d = np.load(_CBM_CONCEPTS_NPZ, allow_pickle=True)
    paths = d["paths"].tolist()
    X = d["X"].astype("float32")
    return {Path(p).name: X[i] for i, p in enumerate(paths)}, X.shape[1]


def load_concept_table_v2():
    """v2: cbm_concepts_v2.npz (11 concepts: SEG 6 + GRAPE-trained RNFL 5,
    for GAMMA_train+REFUGE+GRAPE, n=758). See feature_extraction/extract_concepts_v2.py."""
    d = np.load(_CBM_CONCEPTS_V2_NPZ, allow_pickle=True)
    paths = d["paths"].tolist()
    X = d["X"].astype("float32")
    return {Path(p).name: X[i] for i, p in enumerate(paths)}, X.shape[1]


def grape_baseline_rows():
    """Loads the GRAPE Baseline sheet (one first-visit image per patient, all
    glaucoma). Reused by both the 5-concept RNFL training (fit_rnfl_pls.py)
    and the v2 classifier pool."""
    import local_config as _lc
    root = _lc.GRAPE_ROOT
    df = pd.read_excel(root / "VF and clinical information.xlsx",
                       sheet_name="Baseline", header=None, skiprows=2)
    out = []
    for fn in df[16]:
        if pd.isna(fn):
            continue
        p = root / "CFPs" / str(fn).strip()
        if p.exists():
            out.append((str(p), 1, "GRAPE"))
    return out


def build_pool_v2(seed=42):
    """Builds the unified pool for the v2 classifier: GAMMA_train(100) +
    REFUGE train(400) + GRAPE baseline(263) combined into one DataFrame
    (unlike build_frames(), which only uses GAMMA_train as val to avoid
    leakage, here val is also stratified-sampled from this whole pool -
    2026-08-09 per user request). To preserve glaucoma_ratio, the actual
    split is done by the caller via StratifiedKFold(pool, pool["label"])."""
    gamma = pd.DataFrame(_gamma_train(), columns=["path", "label", "dataset", "case_id"])
    gamma = gamma[["path", "label", "dataset"]]
    refuge = pd.DataFrame(_refuge(), columns=["path", "label", "dataset"])
    grape = pd.DataFrame(grape_baseline_rows(), columns=["path", "label", "dataset"])
    pool = pd.concat([gamma, refuge, grape], ignore_index=True)
    return pool.sample(frac=1, random_state=seed).reset_index(drop=True)


def _concept_vecs(df, concept_table, n_concepts, key_col="path", key_fn=None):
    """Looks up each row's concept vector from concept_table, filling missing entries with zeros."""
    zero_raw = np.zeros(n_concepts, dtype="float32")
    key_fn = key_fn or (lambda p: Path(p).name)
    return np.stack([concept_table.get(key_fn(v), zero_raw) for v in df[key_col]])


def attach_concepts(train_df, val_df, concept_table, n_concepts, return_stats=False):
    """Attaches concept vectors keyed on train_df/val_df["path"] filenames.
    The normalization mean/std are computed only on train_df and applied
    as-is to val_df (and to any third set via the mean/std returned when
    return_stats=True, e.g. GAMMA test) - this prevents fold val stats from
    leaking into the normalization params (earlier 5-fold experiments
    normalized with the full npz, subtly leaking val info).
    Missing entries are filled with the train mean (a zero vector after
    normalization)."""
    Xtr = _concept_vecs(train_df, concept_table, n_concepts)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0

    train_df = train_df.copy()
    train_df["_concept_vec"] = list((Xtr - mean) / std)
    val_df = val_df.copy()
    Xva = _concept_vecs(val_df, concept_table, n_concepts)
    val_df["_concept_vec"] = list((Xva - mean) / std)
    if return_stats:
        return train_df, val_df, mean, std
    return train_df, val_df


def attach_concepts_external(df, concept_table, n_concepts, mean, std,
                             key_col="case_id"):
    """Applies the mean/std from a train fold as-is to attach concepts to a
    third set (e.g. GAMMA test, labels hidden). Defaults key_col to
    "case_id" for cases where concept_table's keys are case_id (extension-
    less strings, as saved by extract_concepts_gamma_test.py)."""
    X = _concept_vecs(df, concept_table, n_concepts, key_col=key_col,
                      key_fn=lambda v: v)
    df = df.copy()
    df["_concept_vec"] = list((X - mean) / std)
    return df


class FundusDS(Dataset):
    """Fundus image dataset; applies train/eval transforms and optionally attaches concept vectors."""

    def __init__(self, df, img_size=224, train=False, with_label=True, use_concepts=False):
        self.df = df.reset_index(drop=True)
        self.with_label = with_label
        self.use_concepts = use_concepts
        if train:
            # VerticalFlip removed: fundus photos are vertically asymmetric
            # (ISNT rule, disc-fovea relative position) which is a key
            # signal for glaucoma detection, and flipping top-bottom would
            # destroy that signal. HorizontalFlip (left/right eye symmetry)
            # is clinically fine and kept.
            # RandomResizedCrop range also narrowed 0.8->0.9 to prevent the
            # disc from being cropped out too aggressively.
            self.tf = transforms.Compose([
                transforms.Lambda(_letterbox_square),
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.15, 0.15, 0.1),
                transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Lambda(_letterbox_square),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
            ])

    def __len__(self):
        """Number of samples."""
        return len(self.df)

    def __getitem__(self, i):
        """Loads and transforms one image, returning (img[, concept], label_or_case_id)."""
        r = self.df.iloc[i]
        img = self.tf(Image.open(r["path"]).convert("RGB"))
        if self.use_concepts:
            import torch
            concept = torch.from_numpy(r["_concept_vec"])
            if self.with_label:
                return img, concept, int(r["label"])
            return img, concept, r.get("case_id", str(i))
        if self.with_label:
            return img, int(r["label"])
        return img, r.get("case_id", str(i))


def class_weights(df):
    """Ratio of negative to positive samples, used as pos_weight for BCE loss."""
    n = (df["label"] == 0).sum()
    p = (df["label"] == 1).sum()
    return float(n / max(1, p))


def loaders(ext, val, batch_size=64, img_size=224, num_workers=4, use_concepts=False):
    """Builds train/val DataLoaders from the ext/val frames."""
    # On Windows, persistent_workers=False (the default) recreates workers
    # every epoch, which effectively hangs with a "Couldn't open shared
    # event" error. persistent_workers=True reuses workers across epochs
    # and fixes this.
    kw = dict(num_workers=num_workers, pin_memory=True,
             persistent_workers=num_workers > 0)
    tr = DataLoader(FundusDS(ext, img_size, train=True, use_concepts=use_concepts),
                    batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    va = DataLoader(FundusDS(val, img_size, train=False, use_concepts=use_concepts),
                    batch_size=batch_size, shuffle=False, **kw)
    return tr, va


if __name__ == "__main__":
    ext, val, test = build_frames()
    print("External (train):", len(ext), ext["dataset"].value_counts().to_dict())
    print("  Labels:", ext["label"].value_counts().to_dict(), "| pos_weight=%.2f" % class_weights(ext))
    print("GAMMA train (val):", len(val), val["label"].value_counts().to_dict())
    print("GAMMA test (target):", len(test))
