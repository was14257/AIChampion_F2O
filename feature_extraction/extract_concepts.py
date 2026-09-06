import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from retfound_seg.model import build_model
from bscan_gen.utils import load_retfound_encoder
from glaucoma_cls.eyeon_cbm import EyeonCBM
from glaucoma_cls.concepts import ALL_CONCEPTS
from glaucoma_cls.concepts import OCT_CONCEPTS
from feature_extraction.datasets_glaucoma import all_pairs

OUT = CFG.paths.oct_features / "cbm_concepts.npz"


def build_cbm(device: str) -> EyeonCBM:
    """Load the frozen segmentation and OCT-linear encoders and assemble the CBM."""
    seg = build_model(load_weights=True)
    ck = torch.load(CFG.paths.ckpt_dir / "best.pth", map_location="cpu", weights_only=False)
    seg.load_state_dict(ck["model"])
    seg.eval().to(device)

    oct_enc = load_retfound_encoder()
    oct_enc.eval().to(device)

    oct_ck = torch.load(CFG.paths.oct_features / "oct_linear.pt",
                        map_location="cpu", weights_only=False)
    oct_linear = nn.Linear(oct_ck["in_dim"], len(OCT_CONCEPTS))
    oct_linear.load_state_dict(oct_ck["state_dict"])
    oct_linear.eval().to(device)

    return EyeonCBM(seg, oct_enc, oct_linear).to(device)


def main():
    """Extract and cache the 6 concepts for all G1020+ORIGA+GAMMA images."""
    device = CFG.runtime.device
    cbm = build_cbm(device)
    pairs = all_pairs()

    rows, labels, paths, ok = [], [], [], []
    for path, label in tqdm(pairs, desc="Extracting concepts"):
        try:
            out = cbm.predict_from_path(path, device=device)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        rows.append([out[c] for c in ALL_CONCEPTS])
        labels.append(label)
        paths.append(path)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    nan_mask = np.isnan(X).any(axis=1)
    print(f"Extracted {len(X)}, excluding {nan_mask.sum()} containing NaN")
    X, y = X[~nan_mask], y[~nan_mask]
    paths = [p for p, m in zip(paths, ~nan_mask) if m]

    np.savez(OUT, X=X, y=y, paths=np.array(paths), concepts=np.array(ALL_CONCEPTS))
    print(f"n={len(X)}  concepts={ALL_CONCEPTS}  -> {OUT}")
    print(f"Glaucoma distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    main()
