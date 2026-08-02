import sys
from pathlib import Path

import numpy as np
import timm
import torch
from scipy.ndimage import gaussian_filter
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG

_RETFOUND_TF = None


def gamma_slice_path(cid: str, si, root: Path | None = None):
    root = root or CFG.paths.gamma_grading
    for s in ("training", "testing"):
        p = Path(root) / f"{s}/multi-modality_images/{cid}/{cid}/{si}_image.jpg"
        if p.exists():
            return p
    return None


def gamma_fundus_path(cid: str, root: Path | None = None):
    root = root or CFG.paths.gamma_grading
    for s in ("training", "testing"):
        p = Path(root) / f"{s}/multi-modality_images/{cid}/{cid}.jpg"
        if p.exists():
            return p
    return None


def pick_per_volume(files, n_per_vol):
    """볼륨(cid)별로 균등 간격 슬라이스 n_per_vol개만 골라 인접 중복을 줄인다."""
    if not n_per_vol:
        return files
    groups = {}
    for f in files:
        cid, si = f.stem.split("_")
        groups.setdefault(cid, []).append((int(si), f))
    picked = []
    for cid, items in groups.items():
        items.sort(key=lambda t: t[0])
        if len(items) <= n_per_vol:
            picked.extend(f for _, f in items)
        else:
            idx = np.linspace(0, len(items) - 1, n_per_vol).astype(int)
            picked.extend(items[i][1] for i in idx)
    return sorted(picked, key=lambda f: f.stem)


def qc_ok(d, thickness_range=(40, 220), max_jump=40) -> bool:
    th = np.median(d["rpe"] - d["ilm"])
    jump = max(np.max(np.abs(np.diff(d["ilm"]))), np.max(np.abs(np.diff(d["rpe"]))))
    lo, hi = thickness_range
    return lo < th < hi and jump < max_jump


def flatten_shift(rpe):
    x = np.arange(len(rpe))
    trend = np.polyval(np.polyfit(x, rpe, 1), x)
    return (trend - trend.mean()).astype(np.float32)


def warp_flatten(img, shift_rows):
    H, W = img.shape
    src = np.clip(np.round(np.arange(H)[:, None] + shift_rows[None, :]).astype(int), 0, H - 1)
    cols = np.broadcast_to(np.arange(W), (H, W))
    return img[src, cols]


def build_cond(ilm, rpe, H=None, W=None, horig=None):
    ot = CFG.oct_tier1
    H = H or ot.diff_h
    W = W or ot.diff_w
    horig = horig or ot.horig
    xs = np.linspace(0, len(ilm) - 1, W)
    il = np.interp(xs, np.arange(len(ilm)), ilm) / horig * H
    rp = np.interp(xs, np.arange(len(rpe)), rpe) / horig * H
    c = np.full((H, W), 0.0, np.float32)
    yy = np.arange(H)[:, None]
    c[(yy >= il[None, :]) & (yy < rp[None, :])] = 1.0
    c[yy >= rp[None, :]] = 0.4
    return c


def load_retfound_encoder(weights_path: Path | None = None):
    weights_path = weights_path or CFG.paths.retfound_weights
    enc = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, img_size=224)
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    st = ck
    for k in ("model", "teacher", "state_dict"):
        if isinstance(ck, dict) and k in ck:
            st = ck[k]
            break
    clean = {}
    for k, v in st.items():
        nk = k
        for pre in ("module.", "encoder.", "backbone."):
            if nk.startswith(pre):
                nk = nk[len(pre):]
        if nk.startswith(("head", "decoder", "mask_token", "fc_norm")):
            continue
        clean[nk] = v
    enc.load_state_dict(clean, strict=False)
    enc = enc.eval()
    return enc.cuda() if torch.cuda.is_available() else enc


def _retfound_tf():
    global _RETFOUND_TF
    if _RETFOUND_TF is None:
        _RETFOUND_TF = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return _RETFOUND_TF


@torch.no_grad()
def embed_fundus(enc, im):
    dev = next(enc.parameters()).device
    x = _retfound_tf()(im.convert("RGB")).unsqueeze(0).to(dev)
    with torch.autocast("cuda", enabled=dev.type == "cuda"):
        f = enc.forward_features(x)[0]
    return f.float().cpu().numpy()[1:].mean(0)


def disc_crop(img, dx):
    g = np.asarray(img.convert("L"), float)
    H, W = g.shape
    dx = int(np.clip(dx, 0, W - 1))
    band = slice(max(0, dx - W // 20), min(W, dx + W // 20))
    prof = gaussian_filter(g[:, band].mean(1), 15)
    dy = int(np.argmax(prof))
    s = int(0.28 * W)
    return img.crop((max(0, dx - s), max(0, dy - s), min(W, dx + s), min(H, dy + s)))
