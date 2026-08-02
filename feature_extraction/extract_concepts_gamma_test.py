"""Extracts the 9 concept types for GAMMA test's (labels hidden, challenge
target) 100 images and saves them separately in the same format as
cbm_concepts.npz.

extract_concepts.py's all_pairs() only covers labeled GAMMA training, so the
test (target) images' concepts aren't in the cache - since glaucoma_cls's
fundus classification switched to using concepts, the actual deployment
(GAMMA test pseudo-label generation) also needs the same concepts, so we
extract them separately here. y is not saved since there are no labels.
"""
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from feature_extraction.extract_concepts import build_cbm
from glaucoma_cls.concepts import ALL_CONCEPTS

OUT = CFG.paths.oct_features / "cbm_concepts_gamma_test.npz"


def gamma_test_paths():
    root = CFG.paths.gamma_grading / "testing/multi-modality_images"
    return [(str(cd / f"{cd.name}.jpg"), cd.name)
            for cd in sorted(root.iterdir()) if cd.is_dir()]


def main():
    device = CFG.runtime.device
    cbm = build_cbm(device)
    pairs = gamma_test_paths()

    rows, case_ids, paths = [], [], []
    for path, cid in tqdm(pairs, desc="Extracting GAMMA test concepts"):
        try:
            out = cbm.predict_from_path(path, device=device)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        rows.append([out[c] for c in ALL_CONCEPTS])
        case_ids.append(cid)
        paths.append(path)

    X = np.array(rows, dtype=np.float32)
    nan_mask = np.isnan(X).any(axis=1)
    print(f"Extracted {len(X)}, excluding {nan_mask.sum()} containing NaN")
    X = X[~nan_mask]
    paths = [p for p, m in zip(paths, ~nan_mask) if m]
    case_ids = [c for c, m in zip(case_ids, ~nan_mask) if m]

    np.savez(OUT, X=X, paths=np.array(paths), case_ids=np.array(case_ids),
             concepts=np.array(ALL_CONCEPTS))
    print(f"n={len(X)}  -> {OUT}")


if __name__ == "__main__":
    main()
