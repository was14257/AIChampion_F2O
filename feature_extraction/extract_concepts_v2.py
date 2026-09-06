import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from bscan_gen.utils import load_retfound_encoder
from glaucoma_cls.concepts import ALL_CONCEPTS_V2
from glaucoma_cls.eyeon_cbm import EyeonCBM
from retfound_seg.model import build_model as build_seg_model

OUT = CFG.paths.oct_features / "cbm_concepts_v2.npz"
DEVICE = CFG.runtime.device


def gamma_pairs():
    """Build (path, label) pairs from labeled GAMMA training images."""
    root = CFG.paths.gamma_grading
    df = pd.read_excel(root / "training/glaucoma_grading_training_GT.xlsx")
    out = []
    for _, r in df.iterrows():
        cid = f"{int(r['data']):04d}"
        p = root / f"training/multi-modality_images/{cid}/{cid}.jpg"
        if p.exists():
            out.append((str(p), 0 if int(r["non"]) == 1 else 1))
    return out


def refuge_pairs():
    """Build (path, label) pairs from REFUGE train images (label from filename prefix 'g')."""
    d = CFG.paths.data_root / "REFUGE/train/Images"
    return [(str(p), 1 if p.stem.lower().startswith("g") else 0)
            for p in sorted(d.glob("*.jpg"))]


def grape_pairs():
    """Build (path, label) pairs from GRAPE baseline visits; all are glaucoma (label=1)."""
    root = CFG.paths.grape
    df = pd.read_excel(root / "VF and clinical information.xlsx",
                       sheet_name="Baseline", header=None, skiprows=2)
    out = []
    for fn in df[16]:
        if pd.isna(fn):
            continue
        p = root / "CFPs" / str(fn).strip()
        if p.exists():
            out.append((str(p), 1))
    return out


def all_pairs():
    """Combine GAMMA train, REFUGE, and GRAPE pairs for concept extraction."""
    g, r, gr = gamma_pairs(), refuge_pairs(), grape_pairs()
    print(f"GAMMA_train={len(g)}  REFUGE={len(r)}  GRAPE={len(gr)}  total={len(g)+len(r)+len(gr)}")
    return g + r + gr


def build_cbm():
    """Load the frozen segmentation encoder and GRAPE-trained RNFL PLS model, assemble the CBM."""
    seg = build_seg_model(load_weights=True)
    seg_ck = torch.load(CFG.paths.ckpt_dir / "best.pth", map_location="cpu", weights_only=False)
    seg.load_state_dict(seg_ck["model"])
    seg.eval().to(DEVICE)

    oct_enc = load_retfound_encoder()
    oct_enc.eval().to(DEVICE)

    with open(CFG.paths.oct_features / "grape_rnfl_pls.pkl", "rb") as f:
        rnfl_model = pickle.load(f)

    return EyeonCBM(seg, oct_enc, oct_linear=None, rnfl_model=rnfl_model).to(DEVICE)


def main():
    """Extract and cache the 11 (SEG + GRAPE-based RNFL) concepts, replacing cbm_concepts.npz."""
    cbm = build_cbm()
    pairs = all_pairs()

    rows, labels, paths = [], [], []
    for path, label in tqdm(pairs, desc="Extracting 11 concepts"):
        try:
            out = cbm.predict_from_path(path, device=DEVICE)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        rows.append([out[c] for c in ALL_CONCEPTS_V2])
        labels.append(label)
        paths.append(path)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    nan_mask = np.isnan(X).any(axis=1)
    print(f"Extracted {len(X)}, excluding {nan_mask.sum()} containing NaN")
    X, y = X[~nan_mask], y[~nan_mask]
    paths = [p for p, m in zip(paths, ~nan_mask) if m]

    np.savez(OUT, X=X, y=y, paths=np.array(paths), concepts=np.array(ALL_CONCEPTS_V2))
    print(f"n={len(X)}  concepts={ALL_CONCEPTS_V2}  -> {OUT}")
    print(f"Glaucoma distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    main()
