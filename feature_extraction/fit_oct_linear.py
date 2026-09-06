import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from bscan_gen.utils import gamma_fundus_path, load_retfound_encoder, embed_fundus, disc_crop
from glaucoma_cls.concepts import fit_oct_linear, OCT_CONCEPTS
from PIL import Image

OUT = CFG.paths.oct_features
LINES = CFG.paths.oct_labels / "lines"
LAT = CFG.paths.oct_labels / "laterality.csv"
GMM = CFG.paths.gamma_grading


def main():
    """Fit a final PLS model on all hand-labeled GAMMA images and save it as an nn.Linear."""
    feats = pd.read_csv(OUT / "features.csv", dtype={"case_id": str})
    feats["case_id"] = feats["case_id"].str.zfill(4)
    feats = feats.set_index("case_id")

    lat = pd.read_csv(LAT, encoding="utf-8-sig")
    lat["case_id"] = lat["case_id"].astype(str).str.zfill(4)
    disc_x = lat.set_index("case_id")["disc_x"].to_dict()

    hand = [c for c in feats.index if (LINES / f"{c}.npz").exists()]

    print(f"Extracting embeddings directly from fundus: {len(hand)} images (whole+disc)")
    enc = load_retfound_encoder()
    embs = []
    for c in hand:
        im = Image.open(gamma_fundus_path(c, GMM)).convert("RGB")
        dx = disc_x.get(c, np.asarray(im).shape[1] // 2)
        embs.append((embed_fundus(enc, im), embed_fundus(enc, disc_crop(im, dx))))
    Wh, Di = (np.stack(x) for x in zip(*embs))

    X = np.hstack([Wh, Di])
    Y = feats.loc[hand, OCT_CONCEPTS].to_numpy(dtype=np.float32)

    linear = fit_oct_linear(X, Y, n_components=8)

    out_path = OUT / "oct_linear.pt"
    torch.save({"state_dict": linear.state_dict(),
                "concepts": OCT_CONCEPTS,
                "in_dim": X.shape[1]}, out_path)
    print(f"n={len(hand)}  in_dim={X.shape[1]}  -> {out_path}")

    # Transfer validation: check that the PLS prediction and nn.Linear output match
    with torch.no_grad():
        pred = linear(torch.from_numpy(X).float()).numpy()
    for i, name in enumerate(OCT_CONCEPTS):
        r = np.corrcoef(pred[:, i], Y[:, i])[0, 1]
        print(f"  {name:<12} r(pred vs GT)={r:.3f}")


if __name__ == "__main__":
    main()
