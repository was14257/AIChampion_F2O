import random
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG

LINES = CFG.paths.oct_labels / "lines"
BSC = CFG.paths.oct_labels / "bscans"
GMM = CFG.paths.gamma_grading
OUTM = CFG.paths.output_root / "ilmrpe"
OUTM.mkdir(parents=True, exist_ok=True)
CKPT = OUTM / "seg_best.pth"

HS = CFG.oct_tier1.seg_h
W = CFG.oct_tier1.seg_w
HORIG = CFG.oct_tier1.horig


def dbl(i, o):
    """Double conv block: Conv-BN-ReLU twice."""
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
    )


class UNet(nn.Module):
    """U-Net that outputs per-column logits over row position for the ILM and RPE lines."""

    def __init__(self):
        super().__init__()
        self.d1 = dbl(1, 32)
        self.d2 = dbl(32, 64)
        self.d3 = dbl(64, 128)
        self.d4 = dbl(128, 256)
        self.p = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c3 = dbl(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = dbl(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.c1 = dbl(64, 32)
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        """Encoder-decoder forward pass with skip connections."""
        e1 = self.d1(x)
        e2 = self.d2(self.p(e1))
        e3 = self.d3(self.p(e2))
        e4 = self.d4(self.p(e3))
        d = self.c3(torch.cat([self.u3(e4), e3], 1))
        d = self.c2(torch.cat([self.u2(d), e2], 1))
        d = self.c1(torch.cat([self.u1(d), e1], 1))
        return self.out(d)


def soft_argmax(logits):
    """Differentiable row-position estimate from per-row softmax logits."""
    p = F.softmax(logits, dim=2)
    idx = torch.arange(logits.shape[2], device=logits.device).view(1, 1, -1, 1)
    return (p * idx).sum(2)


class DS(torch.utils.data.Dataset):
    """Dataset of B-scan images and their ILM/RPE row-position labels, with optional augmentation."""

    def __init__(self, ids, train):
        self.ids = ids
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """Load one sample, applying flip/shift/brightness/noise augmentation when training."""
        c = self.ids[i]
        img = np.asarray(Image.open(BSC / f"{c}.png").convert("L"))
        d = np.load(LINES / f"{c}.npz")
        ilm = d["ilm"].copy()
        rpe = d["rpe"].copy()
        g = np.asarray(Image.fromarray(img).resize((W, HS), Image.BILINEAR), np.float32) / 255.
        y = np.stack([ilm, rpe]) / HORIG * HS

        if self.train:
            if random.random() < 0.5:
                g = g[:, ::-1].copy()
                y = y[:, ::-1].copy()
            sh = random.uniform(-0.08, 0.08) * HS
            g = np.roll(g, int(sh), 0)
            y = np.clip(y + sh, 0, HS - 1)
            g = np.clip(g * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05), 0, 1)
            g = g + np.random.randn(*g.shape).astype(np.float32) * 0.02

        return torch.from_numpy(g)[None], torch.from_numpy(y.astype(np.float32))


def train():
    """Train the ILM/RPE segmentation U-Net, saving the best checkpoint by validation MAE."""
    epochs = CFG.oct_tier1.seg_epochs
    OUTM.mkdir(parents=True, exist_ok=True)
    ids = [
        f.name[:4] for f in sorted(LINES.glob("*.npz"))
        if not (np.isnan(np.load(f)["ilm"]).any() or np.isnan(np.load(f)["rpe"]).any())
    ]
    random.seed(0)
    random.shuffle(ids)
    nv = max(8, len(ids) // 10)
    va, tr = ids[:nv], ids[nv:]
    print(f"train {len(tr)} / val {len(va)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = UNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = torch.utils.data.DataLoader(DS(tr, True), batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    vl = torch.utils.data.DataLoader(DS(va, False), batch_size=8, num_workers=4)

    best = 1e9
    for ep in range(1, epochs + 1):
        m.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            pd = soft_argmax(m(x))
            loss = F.smooth_l1_loss(pd, y)
            loss.backward()
            opt.step()
        sch.step()

        m.eval()
        es = []
        with torch.no_grad():
            for x, y in vl:
                pd = soft_argmax(m(x.to(dev))).cpu()
                es.append((pd - y).abs().mean().item())
        ve = np.mean(es) * HORIG / HS
        if ep % 10 == 0 or ep == 1:
            print(f"[{ep:3d}/{epochs}] val MAE={ve:.2f}px")
        if ve < best:
            best = ve
            torch.save({"model": m.state_dict(), "val_mae": best}, CKPT)
    print(f"done. best val MAE={best:.2f}px -> {CKPT}")


@torch.no_grad()
def predict():
    """Run the trained segmentation model over volumes to produce pseudo ILM/RPE labels and QC overlays."""
    volumes = CFG.oct_tier1.seg_predict_volumes
    slices_per_vol = CFG.oct_tier1.seg_slices_per_vol
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = UNet().to(dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"])
    m.eval()

    outdir = OUTM / "pseudo"
    (outdir / "lines").mkdir(parents=True, exist_ok=True)
    (outdir / "overlays").mkdir(exist_ok=True)
    vols = sorted([
        p for p in GMM.glob("**/multi-modality_images/*/*")
        if p.is_dir() and list(p.glob("*_image.jpg"))
    ])
    if volumes:
        vols = [v for v in vols if v.name in volumes]

    n = 0
    for v in vols:
        cid = v.name
        sl = sorted(
            v.glob("*_image.jpg"),
            key=lambda p: int(p.name.split("_")[0]),
        )
        if not sl:
            continue
        if slices_per_vol:
            idx = np.linspace(0, len(sl) - 1, slices_per_vol).astype(int)
            sl = [sl[i] for i in idx]

        for sp in sl:
            si = sp.name.split("_")[0]
            arr = np.asarray(Image.open(sp).convert("L"))
            g = np.asarray(Image.fromarray(arr).resize((W, HS), Image.BILINEAR), np.float32) / 255.
            pd = soft_argmax(m(torch.from_numpy(g)[None, None].to(dev)))[0].cpu().numpy()
            ilm = pd[0] / HS * HORIG
            rpe = pd[1] / HS * HORIG
            th = np.median(rpe - ilm)
            qc = 40 < th < 220
            key = f"{cid}_{si}"
            np.savez(outdir / "lines" / f"{key}.npz", ilm=ilm, rpe=rpe, width=W, qc=qc)

            if n % 50 == 0:
                full = np.asarray(Image.open(sp).convert("L"))
                plt.figure(figsize=(6, 7))
                plt.imshow(full, cmap="gray", aspect="auto")
                sx = full.shape[1] / W
                plt.plot(np.arange(W) * sx, ilm * full.shape[0] / HORIG, "c-", lw=1)
                plt.plot(np.arange(W) * sx, rpe * full.shape[0] / HORIG, "r-", lw=1)
                plt.title(f"{key} th={th:.0f} qc={qc}")
                plt.axis("off")
                plt.savefig(outdir / "overlays" / f"{key}.png", dpi=80, bbox_inches="tight")
                plt.close()
            n += 1
    print(f"{n} pseudo-labels -> {outdir}/lines  (overlay samples -> overlays)")


if __name__ == "__main__":
    train() if CFG.oct_tier1.seg_run_mode == "train" else predict()
