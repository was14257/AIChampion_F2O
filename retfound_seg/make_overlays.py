from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from config import CFG
from local_config import OVERLAY_SPLIT, OVERLAY_LIMIT, OVERLAY_OUT
from retfound_seg.cdr import compute_cdr, postprocess_label
from retfound_seg.model import build_model


def _preprocess(img_path, device):
    size = CFG.data.img_size
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CFG.data.mean).view(3, 1, 1)
    std = torch.tensor(CFG.data.std).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


@torch.no_grad()
def predict_label(model, img_tensor):
    logits = model(img_tensor)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return postprocess_label(pred)


def _boundary(binary, thickness=3):
    eroded = ndimage.binary_erosion(binary, iterations=max(1, thickness))
    return binary & ~eroded


def save_contour_overlay(img_path, label_map, out_path, thickness=3):
    orig = Image.open(img_path).convert("RGB")
    W, H = orig.size
    rgb = np.asarray(orig).copy()
    lab = np.asarray(Image.fromarray(label_map.astype(np.uint8)).resize((W, H), Image.NEAREST))
    disc = lab >= CFG.data.label_disc_rim
    cup = lab >= CFG.data.label_cup
    t = max(2, round(thickness * max(W, H) / 224))
    rgb[_boundary(disc, t)] = [0, 220, 0]
    rgb[_boundary(cup, t)] = [255, 30, 30]
    Image.fromarray(rgb).save(out_path)


def main():
    device = torch.device(CFG.runtime.device)
    model = build_model(load_weights=False).to(device)
    ckpt = torch.load(CFG.infer.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"[overlay] loaded (mean_dice={ckpt.get('mean_dice')})")

    img_dir = {"train": CFG.paths.refuge_train_img,
               "val": CFG.paths.refuge_val_img,
               "test": CFG.paths.refuge_test_img}[OVERLAY_SPLIT]
    out_dir = Path(OVERLAY_OUT) if OVERLAY_OUT else CFG.paths.pred_dir / f"{OVERLAY_SPLIT}_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("*.jpg"))
    if OVERLAY_LIMIT:
        imgs = imgs[:OVERLAY_LIMIT]
    for i, p in enumerate(imgs, 1):
        pred = predict_label(model, _preprocess(p, device))
        cdr = compute_cdr(pred)
        save_contour_overlay(p, pred, out_dir / f"{p.stem}_overlay.png")
        if i % 25 == 0 or i == len(imgs):
            print(f"  {i}/{len(imgs)}  {p.stem} CDR={cdr:.3f}")
    print(f"\nDone: {len(imgs)} images -> {out_dir}")


if __name__ == "__main__":
    main()
