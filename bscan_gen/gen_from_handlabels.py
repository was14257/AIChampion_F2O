import random
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bscan_gen import diffusion_bscan as D

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG

LINES = CFG.paths.oct_labels / "lines"
BSC = CFG.paths.oct_labels / "bscans"
OUTG = D.OUT / "handlabel_gen"
OUTG.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def ddim_sample(m, conds, steps=250):
    dev = conds.device
    b, a, ac = D.make_sched(dev)
    n = conds.size(0)
    x = torch.randn(n, 1, D.H, D.W, device=dev)
    ts = np.linspace(0, D.T - 1, steps).astype(int)[::-1]

    for k, i in enumerate(ts):
        t = torch.full((n,), int(i), device=dev, dtype=torch.long)
        with torch.autocast("cuda"):
            eps = m(x, conds, t)
        aci = ac[i]
        x0 = ((x - (1 - aci).sqrt() * eps) / aci.sqrt()).clamp(-1, 1)
        if k < len(ts) - 1:
            ai = ac[int(ts[k + 1])]
            x = ai.sqrt() * x0 + (1 - ai).sqrt() * eps
        else:
            x = x0
    return x


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = D.UNet().to(dev)
    m.load_state_dict(torch.load(D.CKPT, map_location=dev, weights_only=False)["model"])
    m.eval()

    files = [
        f for f in sorted(LINES.glob("*.npz"))
        if not (np.isnan(np.load(f)["ilm"]).any() or np.isnan(np.load(f)["rpe"]).any())
    ]
    print(f"Generating {len(files)} hand-labeled cases...")
    ids = [f.stem for f in files]

    def cond_of(f):
        d = np.load(f)
        ilm, rpe = d["ilm"], d["rpe"]
        if D.FLATTEN:
            shift = D.flatten_shift(rpe)
            ilm, rpe = ilm - shift, rpe - shift
        return D.build_cond(ilm, rpe)

    conds = torch.stack([torch.from_numpy(cond_of(f))[None] for f in files])

    gens = []
    for s in range(0, len(files), 16):
        c = conds[s:s + 16].to(dev)
        g = ddim_sample(m, c, steps=100)
        gens.append(((g.clamp(-1, 1) + 1) * 127.5).cpu().numpy()[:, 0])
        print(f"  {min(s + 16, len(files))}/{len(files)}")
    gens = np.concatenate(gens)

    for cid, g in zip(ids, gens):
        Image.fromarray(g.astype(np.uint8)).save(OUTG / f"{cid}_gen.png")
    print(f"generation saved -> {OUTG}")

    random.seed(0)
    pick = random.sample(range(len(ids)), 12)
    fig, ax = plt.subplots(3, 12, figsize=(24, 6.5))
    for j, idx in enumerate(pick):
        cid = ids[idx]
        real = np.asarray(Image.open(BSC / f"{cid}.png").convert("L").resize((D.W, D.H)))
        ax[0, j].imshow(conds[idx, 0], cmap="viridis", aspect="auto")
        ax[0, j].axis("off")
        ax[0, j].set_title(cid, fontsize=8)
        ax[1, j].imshow(gens[idx], cmap="gray", aspect="auto")
        ax[1, j].axis("off")
        ax[2, j].imshow(real, cmap="gray", aspect="auto")
        ax[2, j].axis("off")
    for r, lab in zip(range(3), ["cond", "GEN", "real"]):
        ax[r, 0].set_ylabel(lab)
    fig.suptitle("Hand-label sketch -> diffusion.  row1 cond / row2 GENERATED / row3 real central B-scan")
    plt.tight_layout()
    plt.savefig(D.OUT / "handlabel_compare.png", dpi=95)
    plt.close()
    print(f"comparison grid -> {D.OUT / 'handlabel_compare.png'}")


if __name__ == "__main__":
    main()
