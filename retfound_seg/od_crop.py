from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CFG
from local_config import ODCROP_MARGIN, ODCROP_OUT_SIZE
from retfound_seg.model import build_model


def load_seg_model(ckpt_path: Path, device: torch.device):
    """Load a trained segmentation model checkpoint."""
    model = build_model(load_weights=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@torch.no_grad()
def predict_disc_mask(model, img_pil: Image.Image, seg_size: int,
                      device: torch.device) -> np.ndarray:
    """Predict the optic disc binary mask for an image, resized back to original size."""
    orig_w, orig_h = img_pil.size

    img_resized = img_pil.resize((seg_size, seg_size), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img_resized, dtype=np.float32) / 255.0)
    t = t.permute(2, 0, 1)
    t = (t - _MEAN) / _STD
    t = t.unsqueeze(0).to(device)

    logits = model(t)
    pred = logits.argmax(dim=1)[0]

    disc_mask = (pred >= CFG.data.label_disc_rim).cpu().numpy().astype(np.uint8)

    disc_pil = Image.fromarray(disc_mask * 255).resize(
        (orig_w, orig_h), Image.NEAREST
    )
    return np.asarray(disc_pil) > 0


def disc_bbox_crop(img_pil: Image.Image, disc_mask: np.ndarray,
                   margin: float = 1.5) -> Image.Image:
    """Crop a square region around the disc mask (with margin), or center-crop as fallback."""
    w, h = img_pil.size
    ys, xs = np.where(disc_mask)

    if len(xs) == 0:
        side = min(w, h) // 2
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return img_pil.crop((x1, y1, x2, y2))

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    r  = max(x_max - x_min, y_max - y_min) / 2

    side = int(r * 2 * margin)
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    if x2 - x1 < side:
        x1 = max(0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0, y2 - side)

    return img_pil.crop((x1, y1, x2, y2))


def crop_all(margin: float = 1.5, out_size: int = 448):
    """Run disc detection + cropping over all GRAPE CFP images and save the results."""
    device = torch.device(CFG.runtime.device)
    seg_size = CFG.data.img_size
    ckpt_path = CFG.infer.checkpoint_path

    print(f"Segmentation checkpoint: {ckpt_path}")
    model = load_seg_model(ckpt_path, device)

    cfp_dir  = CFG.paths.grape / "CFPs"
    out_dir  = CFG.paths.grape / "CFPs_cropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(cfp_dir.glob("*.jpg")) + sorted(cfp_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"No GRAPE CFP images found: {cfp_dir}")

    print(f"Images to process: {len(imgs)}  ->  {out_dir}")

    fallback_cnt = 0
    for i, p in enumerate(imgs, 1):
        img_pil = Image.open(p).convert("RGB")
        disc_mask = predict_disc_mask(model, img_pil, seg_size, device)

        if disc_mask.sum() == 0:
            fallback_cnt += 1

        cropped = disc_bbox_crop(img_pil, disc_mask, margin=margin)
        cropped = cropped.resize((out_size, out_size), Image.BILINEAR)

        out_path = out_dir / p.name
        cropped.save(out_path)

        if i % 20 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}] {p.name}  disc_px={disc_mask.sum()}")

    print(f"\nDone: {len(imgs)} images saved -> {out_dir}")
    if fallback_cnt:
        print(f"  Warning: disc not detected, fallback for {fallback_cnt} images (center crop applied)")


if __name__ == "__main__":
    crop_all(margin=ODCROP_MARGIN, out_size=ODCROP_OUT_SIZE)
