import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from config import CFG
from local_config import INFER_MODE, INFER_SPLIT, INFER_IMAGE, INFER_SAVE_VIS
from retfound_seg.cdr import compute_cdr, postprocess_label
from retfound_seg.dataset import build_loader
from retfound_seg.model import build_model

_PALETTE = np.array([[0, 0, 0], [0, 170, 0], [220, 0, 0]], dtype=np.uint8)


def load_model(device):
    """Load the trained segmentation model checkpoint for inference."""
    ckpt_path = CFG.infer.checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun `python train.py` first to train.")
    model = build_model(load_weights=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[infer] loaded {ckpt_path} (mean_dice={ckpt.get('mean_dice')})")
    return model


def _preprocess(img_path: Path, device):
    """Load an image file and prepare it as a normalized model input tensor."""
    size = CFG.data.img_size
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CFG.data.mean).view(3, 1, 1)
    std = torch.tensor(CFG.data.std).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def predict_label(model, img_tensor) -> np.ndarray:
    """Run the model and return a post-processed label map."""
    logits = model(img_tensor)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return postprocess_label(pred)


def save_overlay(label_map: np.ndarray, out_path: Path):
    """Save a label map as a color-palette image."""
    Image.fromarray(_PALETTE[label_map]).save(out_path)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom


def _boundary(binary: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Return the outline pixels of a binary mask via erosion."""
    eroded = ndimage.binary_erosion(binary, iterations=max(1, thickness))
    return binary & ~eroded


def save_contour_overlay(img_path: Path, label_map: np.ndarray, out_path: Path,
                         thickness: int = 3):
    """Draw disc/cup contours on top of the original image and save it."""
    orig = Image.open(img_path).convert("RGB")
    W, H = orig.size
    rgb = np.asarray(orig).copy()

    lab = np.asarray(
        Image.fromarray(label_map.astype(np.uint8)).resize((W, H), Image.NEAREST)
    )

    disc = lab >= CFG.data.label_disc_rim
    cup = lab >= CFG.data.label_cup
    t = max(2, round(thickness * max(W, H) / 224))

    rgb[_boundary(disc, t)] = [0, 220, 0]
    rgb[_boundary(cup, t)] = [255, 30, 30]
    Image.fromarray(rgb).save(out_path)


def run_single(image: str):
    """Run inference on a single image and print/save the CDR result."""
    device = torch.device(CFG.runtime.device)
    model = load_model(device)
    img_path = Path(image)
    pred = predict_label(model, _preprocess(img_path, device))
    cdr = compute_cdr(pred)
    CFG.paths.ensure_dirs()
    out = CFG.paths.pred_dir / f"{img_path.stem}_contour.png"
    save_contour_overlay(img_path, pred, out)
    print(f"\nImage: {img_path.name}")
    print(f"Predicted C/D ratio ({CFG.infer.cdr_kind}) = {cdr:.4f}")
    print(f"Contour overlay saved: {out}")


def run_split(split: str, save_vis: bool = False):
    """Run inference over a full dataset split, saving a CDR/dice CSV summary."""
    device = torch.device(CFG.runtime.device)
    model = load_model(device)
    loader = build_loader(split, batch_size=1, shuffle=False)
    CFG.paths.ensure_dirs()

    img_dir = {"train": CFG.paths.refuge_train_img,
               "val": CFG.paths.refuge_val_img,
               "test": CFG.paths.refuge_test_img}[split]
    vis_dir = CFG.paths.pred_dir / f"{split}_contours"
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pred_cdrs, gt_cdrs = [], []
    disc_dices, cup_dices = [], []
    dl, cl = CFG.data.label_disc_rim, CFG.data.label_cup
    for batch in loader:
        img = batch["image"].to(device)
        gt = batch["mask"][0].numpy().astype(np.int64)
        stem = batch["stem"][0]

        pred = predict_label(model, img)
        pc = compute_cdr(pred)
        gc = compute_cdr(gt)

        disc_d = _dice(pred >= dl, gt >= dl)
        cup_d = _dice(pred >= cl, gt >= cl)
        disc_dices.append(disc_d)
        cup_dices.append(cup_d)

        rows.append({"image": stem, "pred_cdr": pc, "gt_cdr": gc,
                     "disc_dice": disc_d, "cup_dice": cup_d})
        if np.isfinite(pc) and np.isfinite(gc):
            pred_cdrs.append(pc)
            gt_cdrs.append(gc)

        if save_vis:
            save_contour_overlay(img_dir / f"{stem}.jpg", pred,
                                 vis_dir / f"{stem}_contour.png")

    csv_path = CFG.paths.pred_dir / f"cdr_{split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "pred_cdr", "gt_cdr", "disc_dice", "cup_dice"])
        w.writeheader()
        w.writerows(rows)

    pred_cdrs, gt_cdrs = np.array(pred_cdrs), np.array(gt_cdrs)
    mae = np.mean(np.abs(pred_cdrs - gt_cdrs))
    corr = np.corrcoef(pred_cdrs, gt_cdrs)[0, 1] if len(pred_cdrs) > 1 else float("nan")
    disc_dice = float(np.mean(disc_dices)) if disc_dices else float("nan")
    cup_dice = float(np.mean(cup_dices)) if cup_dices else float("nan")
    mean_dice = 0.5 * (disc_dice + cup_dice)
    print(f"\n[{split}] n={len(rows)}")
    print(f"  Dice  disc={disc_dice:.4f}  cup={cup_dice:.4f}  mean={mean_dice:.4f}")
    print(f"  C/D ratio({CFG.infer.cdr_kind})")
    print(f"  MAE (pred vs GT-mask) = {mae:.4f}")
    print(f"  Pearson r            = {corr:.4f}")
    print(f"  Result CSV: {csv_path}")
    if save_vis:
        print(f"  Contour overlays: {vis_dir}")


def main():
    """Entry point: run single-image or full-split inference based on config."""
    if INFER_MODE == "image":
        run_single(INFER_IMAGE)
    else:
        run_split(INFER_SPLIT, save_vis=INFER_SAVE_VIS)


if __name__ == "__main__":
    main()
