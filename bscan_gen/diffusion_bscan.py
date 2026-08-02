import math
import random
import sys
import warnings
from dataclasses import dataclass
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
from bscan_gen.utils import (gamma_slice_path, qc_ok as _qc_ok, build_cond as _build_cond,
                             flatten_shift, warp_flatten, pick_per_volume)

GMM = CFG.paths.gamma_grading
PSE = CFG.paths.oct_pseudo / "lines"
OUT = CFG.paths.diffusion_out
OUT.mkdir(parents=True, exist_ok=True)

H = CFG.oct_tier1.diff_h
W = CFG.oct_tier1.diff_w
HORIG = CFG.oct_tier1.horig
T = CFG.oct_tier1.diff_t
FLATTEN = CFG.oct_tier1.flatten_rpe
CKPT = OUT / f"ddpm{H}{'_flat' if FLATTEN else ''}.pth"


@dataclass(frozen=True)
class DiffusionSpec:
    """Resolution a diffusion checkpoint was trained at, paired with its weights.

    Callers used to select the 512 model by overwriting this module's global
    H/W/CKPT at runtime (fundus_to_oct_e2e.py). That made the same function
    behave differently depending on which script imported it first: running
    gen_from_handlabels.py directly gave 256, reaching it through the e2e
    script gave 512. Passing the resolution in removes that hidden coupling."""
    h: int
    w: int
    ckpt: Path
    horig: int = HORIG
    t: int = T
    flatten: bool = FLATTEN


def default_spec() -> DiffusionSpec:
    """Resolution/checkpoint the config points at (currently 256)."""
    return DiffusionSpec(h=H, w=W, ckpt=CKPT)


def spec_512() -> DiffusionSpec:
    """The 512 model used by the deployed demo and e2e generation."""
    return DiffusionSpec(h=512, w=512, ckpt=OUT / "ddpm512_flat.pth")


def load_pair(cid, si, ilm, rpe):
    img = np.asarray(Image.open(slice_path(cid, si)).convert("L").resize((W, H)), np.float32)
    if FLATTEN:
        shift = flatten_shift(rpe)
        ilm = ilm - shift
        rpe = rpe - shift
        img = warp_flatten(img, np.interp(np.linspace(0, len(shift) - 1, W),
                                          np.arange(len(shift)), shift) / HORIG * H)
    return img, ilm, rpe


def slice_path(cid, si):
    return gamma_slice_path(cid, si, GMM)


def qc_ok(d):
    return _qc_ok(d)


def build_cond(ilm, rpe):
    return _build_cond(ilm, rpe, H, W, HORIG)


class DS(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        d = np.load(f)
        cid, si = f.stem.split("_")
        img, ilm, rpe = load_pair(cid, si, d["ilm"], d["rpe"])
        img = img / 127.5 - 1
        cond = build_cond(ilm, rpe)
        return torch.from_numpy(img)[None], torch.from_numpy(cond)[None]


def tpe(t, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freq[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Res(nn.Module):

    def __init__(self, ci, co, td):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci)
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.emb = nn.Linear(td, co)
        self.n2 = nn.GroupNorm(8, co)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(t)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.sk(x)


class UNet(nn.Module):

    def __init__(self, ch=64, td=256):
        super().__init__()
        self.td = td
        self.tmlp = nn.Sequential(nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.inc = nn.Conv2d(2, ch, 3, padding=1)
        self.d1 = Res(ch, ch, td)
        self.d2 = Res(ch, ch * 2, td)
        self.d3 = Res(ch * 2, ch * 4, td)
        self.d4 = Res(ch * 4, ch * 4, td)
        self.pool = nn.AvgPool2d(2)
        self.mid = Res(ch * 4, ch * 4, td)
        self.u4 = Res(ch * 4 + ch * 4, ch * 4, td)
        self.u3 = Res(ch * 4 + ch * 4, ch * 2, td)
        self.u2 = Res(ch * 2 + ch * 2, ch, td)
        self.u1 = Res(ch + ch, ch, td)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, cond, t):
        te = self.tmlp(tpe(t, self.td))
        h = self.inc(torch.cat([x, cond], 1))
        e1 = self.d1(h, te)
        e2 = self.d2(self.pool(e1), te)
        e3 = self.d3(self.pool(e2), te)
        e4 = self.d4(self.pool(e3), te)
        m = self.mid(self.pool(e4), te)
        u = F.interpolate(m, scale_factor=2)
        u = self.u4(torch.cat([u, e4], 1), te)
        u = F.interpolate(u, scale_factor=2)
        u = self.u3(torch.cat([u, e3], 1), te)
        u = F.interpolate(u, scale_factor=2)
        u = self.u2(torch.cat([u, e2], 1), te)
        u = F.interpolate(u, scale_factor=2)
        u = self.u1(torch.cat([u, e1], 1), te)
        return self.out(u)


def make_sched(dev, t_steps=None):
    b = torch.linspace(1e-4, 0.02, t_steps or T, device=dev)
    a = 1 - b
    ac = torch.cumprod(a, 0)
    return b, a, ac


@torch.no_grad()
def ddim_sample(m, conds, spec, steps=100, progress_cb=None):
    """Deterministic (eta=0) DDIM sampling at the resolution given by spec.

    This logic previously existed twice: gen_from_handlabels.ddim_sample read
    the module globals (so it built noise at 256 and mismatched the 512
    checkpoint), and app_streamlit.py kept its own 512 copy. The copy had two
    improvements the original lacked - autocast disabled on CPU, and a
    progress callback - both folded in here.

    progress_cb(k, steps): called per denoising step, for UI progress.
    """
    dev = conds.device
    b, a, ac = make_sched(dev, spec.t)
    n = conds.size(0)
    x = torch.randn(n, 1, spec.h, spec.w, device=dev)
    ts = np.linspace(0, spec.t - 1, steps).astype(int)[::-1]

    for k, i in enumerate(ts):
        t = torch.full((n,), int(i), device=dev, dtype=torch.long)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            eps = m(x, conds, t)
        aci = ac[i]
        x0 = ((x - (1 - aci).sqrt() * eps) / aci.sqrt()).clamp(-1, 1)
        if k < len(ts) - 1:
            ai = ac[int(ts[k + 1])]
            x = ai.sqrt() * x0 + (1 - ai).sqrt() * eps
        else:
            x = x0
        if progress_cb is not None:
            progress_cb(k + 1, steps)
    return x


def load_unet(spec, dev):
    """Load the UNet from spec's checkpoint (the net itself is conv-only, so
    it is resolution-agnostic - only the sampling noise shape depends on spec)."""
    m = UNet().to(dev)
    m.load_state_dict(torch.load(spec.ckpt, map_location=dev, weights_only=False)["model"])
    m.eval()
    return m


def build_cond_for(ilm, rpe, spec):
    """Condition sketch rendered at spec's resolution."""
    return _build_cond(ilm, rpe, spec.h, spec.w, spec.horig)


def train():
    epochs = CFG.oct_tier1.diff_epochs
    bs = CFG.oct_tier1.diff_bs
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    files = [f for f in sorted(PSE.glob("*.npz")) if qc_ok(np.load(f))]
    files = pick_per_volume(files, CFG.oct_tier1.diff_slices_per_vol)
    random.seed(0)
    random.shuffle(files)
    print(f"{len(files)} training pairs")

    dl = torch.utils.data.DataLoader(
        DS(files), batch_size=bs, shuffle=True, num_workers=6,
        drop_last=True, pin_memory=True,
    )
    m = UNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 2e-4)
    b, a, ac = make_sched(dev)
    scaler = torch.cuda.amp.GradScaler()

    nb = len(dl)
    log_every = max(1, nb // 10)
    for ep in range(1, epochs + 1):
        m.train()
        tot = 0
        for bi, (img, cond) in enumerate(dl, 1):
            img, cond = img.to(dev), cond.to(dev)
            t = torch.randint(0, T, (img.size(0),), device=dev)
            noise = torch.randn_like(img)
            act = ac[t][:, None, None, None]
            xt = act.sqrt() * img + (1 - act).sqrt() * noise

            opt.zero_grad()
            with torch.autocast("cuda"):
                pred = m(xt, cond, t)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()

            if bi % log_every == 0 or bi == nb:
                print(f"  ep{ep:3d} [{bi:4d}/{nb}] loss={tot / bi:.4f}", flush=True)

        print(f"ep{ep:3d} done  loss={tot / len(dl):.4f}", flush=True)
        if ep % 5 == 0 or ep == epochs:
            torch.save({"model": m.state_dict()}, CKPT)
            sample(n=6, ep=ep, m=m, files=files)
    print("done")


@torch.no_grad()
def sample(n=None, ep="final", m=None, files=None):
    n = n or CFG.oct_tier1.diff_sample_n
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if m is None:
        m = UNet().to(dev)
        m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"])
        m.eval()
        files = [f for f in sorted(PSE.glob("*.npz")) if qc_ok(np.load(f))]
        random.seed(1)
    m.eval()

    b, a, ac = make_sched(dev)
    picks = random.sample(files, n)
    real, conds_np = [], []
    for f in picks:
        d = np.load(f)
        cid, si = f.stem.split("_")
        img, ilm, rpe = load_pair(cid, si, d["ilm"], d["rpe"])
        real.append(img.astype(np.uint8))
        conds_np.append(build_cond(ilm, rpe))
    conds = torch.stack([torch.from_numpy(c)[None] for c in conds_np]).to(dev)

    x = torch.randn(n, 1, H, W, device=dev)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        with torch.autocast("cuda"):
            eps = m(x, conds, t)
        act = ac[i]
        at = a[i]
        mean = (1 / at.sqrt()) * (x - b[i] / (1 - act).sqrt() * eps)
        x = mean + (b[i].sqrt() * torch.randn_like(x) if i > 0 else 0)

    gen = ((x.clamp(-1, 1) + 1) * 127.5).cpu().numpy()[:, 0]
    cc = conds.cpu().numpy()[:, 0]

    fig, ax = plt.subplots(3, n, figsize=(3 * n, 7))
    for j in range(n):
        ax[0, j].imshow(cc[j], cmap="viridis", aspect="auto")
        ax[0, j].axis("off")
        ax[1, j].imshow(gen[j], cmap="gray", aspect="auto")
        ax[1, j].axis("off")
        ax[2, j].imshow(real[j], cmap="gray", aspect="auto")
        ax[2, j].axis("off")
    ax[0, 0].set_ylabel("cond")
    fig.suptitle("row1=condition  row2=GENERATED  row3=real")
    plt.tight_layout()
    plt.savefig(OUT / f"sample_ep{ep}.png", dpi=90)
    plt.close()
    print(f"  sample saved -> sample_ep{ep}.png")


if __name__ == "__main__":
    train() if CFG.oct_tier1.diff_run_mode == "train" else sample()
