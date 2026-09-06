import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG


def refuge_pairs() -> list[tuple[str, int]]:
    """Build (path, label) pairs from REFUGE train images (label from filename prefix 'g')."""
    d = CFG.paths.data_root / "REFUGE/train/Images"
    return [(str(p), 1 if p.stem.lower().startswith("g") else 0)
            for p in sorted(d.glob("*.jpg"))]


def g1020_pairs() -> list[tuple[str, int]]:
    """Build (path, label) pairs from the G1020 dataset's CSV."""
    root = CFG.paths.g1020
    df = pd.read_csv(root / "G1020.csv")
    return [(str(root / "Images" / row.imageID), int(row.binaryLabels))
            for row in df.itertuples()]


def origa_pairs() -> list[tuple[str, int]]:
    """Build (path, label) pairs from the ORIGA dataset's CSV."""
    root = CFG.paths.origa
    df = pd.read_csv(root / "OrigaList.csv")
    return [(str(root / "Images" / row.Filename), int(row.Glaucoma))
            for row in df.itertuples()]


def gamma_pairs() -> list[tuple[str, int]]:
    """Build (path, label) pairs from the GAMMA grading dataset (training + testing GT sheets)."""
    from bscan_gen.utils import gamma_fundus_path
    root = CFG.paths.gamma_grading
    rows = []
    for split in ("training", "testing"):
        xlsx = root / split / "glaucoma_grading_training_GT.xlsx"
        if not xlsx.exists():
            continue
        gt = pd.read_excel(xlsx)
        for r in gt.itertuples():
            cid = f"{int(r.data):04d}"
            p = gamma_fundus_path(cid, root)
            if p is not None:
                rows.append((str(p), 0 if int(r.non) == 1 else 1))
    return rows


def all_pairs() -> list[tuple[str, int]]:
    """Combine pairs from REFUGE, G1020, ORIGA, and GAMMA for risk-head training."""
    pairs = refuge_pairs() + g1020_pairs() + origa_pairs() + gamma_pairs()
    print(f"REFUGE={len(refuge_pairs())}  G1020={len(g1020_pairs())}  "
          f"ORIGA={len(origa_pairs())}  GAMMA={len(gamma_pairs())}  total={len(pairs)}")
    return pairs


if __name__ == "__main__":
    all_pairs()
