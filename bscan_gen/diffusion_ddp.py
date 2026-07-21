import math
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image, ImageFile

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from utils import (gamma_slice_path, qc_ok, build_cond as _build_cond,
                  flatten_shift, warp_flatten)

HORIG = CFG.oct_tier1.horig
T = CFG.oct_tier1.diff_t
FLATTEN = CFG.oct_tier1.flatten_rpe


def slice_path(root, cid, si):
    return gamma_slice_path(cid, si, root)


def build_cond(ilm, rpe, H, W):
    return _build_cond(ilm, rpe, H, W, HORIG)


class DS(torch.utils.data.Dataset):

    def __init__(self, files, root, H, W):
        self.files = files
        self.root = root
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        d = np.load(f)
        cid, si = f.stem.split("_")
        p = slice_path(self.root, cid, si)
        img = np.asarray(Image.open(p).convert("L").resize((self.W, self.H)), np.float32)
        ilm, rpe = d["ilm"], d["rpe"]
        if FLATTEN:
            shift = flatten_shift(rpe)
            ilm = ilm - shift
            rpe = rpe - shift
            shift_rows = np.interp(np.linspace(0, len(shift) - 1, self.W),
                                   np.arange(len(shift)), shift) / HORIG * self.H
            img = warp_flatten(img, shift_rows)
        img = img / 127.5 - 1
        cond = build_cond(ilm, rpe, self.H, self.W)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            cond = cond[:, ::-1].copy()
        return torch.from_numpy(img)[None], torch.from_numpy(cond)[None]


def tpe(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None]
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

    def __init__(self, ch=128, td=256):
        super().__init__()
        self.td = td
        self.tmlp = nn.Sequential(nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.inc = nn.Conv2d(2, ch, 3, padding=1)
        mult = [1, 2, 4, 4, 8]
        chs = [ch * m for m in mult]

        self.downs = nn.ModuleList()
        prev = ch
        for c in chs:
            self.downs.append(Res(prev, c, td))
            prev = c
        self.pool = nn.AvgPool2d(2)
        self.mid = Res(prev, prev, td)

        self.ups = nn.ModuleList()
        rev = list(reversed(chs))
        prevu = prev
        for i, c in enumerate(rev):
            skip = chs[len(chs) - 1 - i]
            self.ups.append(Res(prevu + skip, c, td))
            prevu = c
        self.out = nn.Sequential(nn.GroupNorm(8, prevu), nn.SiLU(), nn.Conv2d(prevu, 1, 3, padding=1))

    def forward(self, x, cond, t):
        te = self.tmlp(tpe(t, self.td))
        h = self.inc(torch.cat([x, cond], 1))
        skips = []
        for d in self.downs:
            h = d(h, te)
            skips.append(h)
            h = self.pool(h)
        h = self.mid(h, te)
        for u in self.ups:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            sk = skips.pop()
            if h.shape[-2:] != sk.shape[-2:]:
                h = F.interpolate(h, size=sk.shape[-2:], mode="nearest")
            h = u(torch.cat([h, sk], 1), te)
        return self.out(h)


def make_sched(dev):
    b = torch.linspace(1e-4, 0.02, T, device=dev)
    a = 1 - b
    ac = torch.cumprod(a, 0)
    return b, a, ac


def setup_ddp():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def main():
    ot = CFG.oct_tier1
    root, pseudo, out = ot.diff_full_root, ot.diff_full_pseudo, ot.diff_full_out
    epochs = ot.diff_full_epochs
    bs = ot.diff_full_bs
    ch = ot.diff_full_ch
    lr = ot.diff_full_lr
    save_every = ot.diff_full_save_every
    H, W = ot.diff_full_h, ot.diff_full_w
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rank, local_rank, world = setup_ddp()
    dev = torch.device(f"cuda:{local_rank}")

    files = [f for f in sorted(Path(pseudo).glob("*.npz")) if qc_ok(np.load(f))]
    if rank == 0:
        print(f"학습쌍 {len(files)}개, world_size={world}, res={H}x{W}, ch={ch}")

    ds = DS(files, root, H, W)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=bs, sampler=sampler, num_workers=6,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )

    m = UNet(ch=ch).to(dev)
    m = DDP(m, device_ids=[local_rank])
    opt = torch.optim.AdamW(m.parameters(), lr)
    scaler = torch.cuda.amp.GradScaler()
    b, a, ac = make_sched(dev)

    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        m.train()
        tot = 0.0
        n = 0
        for img, cond in dl:
            img = img.to(dev, non_blocking=True)
            cond = cond.to(dev, non_blocking=True)
            t = torch.randint(0, T, (img.size(0),), device=dev)
            noise = torch.randn_like(img)
            act = ac[t][:, None, None, None]
            xt = act.sqrt() * img + (1 - act).sqrt() * noise

            opt.zero_grad()
            with torch.autocast("cuda"):
                pred = m(xt, cond, t)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1

        if rank == 0:
            print(f"ep{ep:3d} loss={tot / n:.4f}")
            if ep % save_every == 0 or ep == epochs:
                torch.save({"model": m.module.state_dict(), "H": H, "W": W, "ch": ch}, out / "ddpm_full.pth")
                print(f"  체크포인트 저장 → {out / 'ddpm_full.pth'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
