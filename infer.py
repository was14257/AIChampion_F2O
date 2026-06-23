"""
학습된 RETFound 세그멘테이션 모델로 C/D ratio 추론.

사용법:
    # REFUGE 한 split 전체에 대해 예측 + GT 와 비교(CSV/PNG 저장)
    python infer.py --split val

    # 단일 이미지 (ROI 크롭본 권장: 디스크 중심 정사각)
    python infer.py --image path/to/disc_crop.jpg

REFUGE 에는 C/D ratio 정답 라벨이 없으므로, GT 마스크에서 계산한 C/D ratio 를
"정답"으로 보고 예측 C/D ratio 와 비교한다(MAE/상관계수).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from local_config import CFG
from retfound_seg.cdr import compute_cdr, postprocess_label
from retfound_seg.dataset import build_loader
from retfound_seg.model import build_model


# 라벨맵 -> 시각화용 컬러 (배경=검정, disc=초록, cup=빨강)
_PALETTE = np.array([[0, 0, 0], [0, 170, 0], [220, 0, 0]], dtype=np.uint8)


def load_model(device):
    ckpt_path = CFG.infer.checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트 없음: {ckpt_path}\n먼저 `python train.py` 로 학습하세요.")
    model = build_model(load_weights=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[infer] loaded {ckpt_path} (mean_dice={ckpt.get('mean_dice')})")
    return model


def _preprocess(img_path: Path, device):
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
    logits = model(img_tensor)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    # 가장 큰 disc 덩어리만 남겨 오탐(밝은 반사광 등) 제거
    return postprocess_label(pred)


def save_overlay(label_map: np.ndarray, out_path: Path):
    Image.fromarray(_PALETTE[label_map]).save(out_path)


def _boundary(binary: np.ndarray, thickness: int = 3) -> np.ndarray:
    """이진 마스크의 테두리(윤곽선) 픽셀을 두께 thickness 로 추출."""
    try:
        from scipy import ndimage
        eroded = ndimage.binary_erosion(binary, iterations=max(1, thickness))
        return binary & ~eroded
    except Exception:
        # scipy 없으면 4-이웃 차이로 1px 경계만
        b = binary
        edge = np.zeros_like(b)
        edge[1:, :] |= b[1:, :] != b[:-1, :]
        edge[:-1, :] |= b[:-1, :] != b[1:, :]
        edge[:, 1:] |= b[:, 1:] != b[:, :-1]
        edge[:, :-1] |= b[:, :-1] != b[:, 1:]
        return edge & b


def save_contour_overlay(img_path: Path, label_map: np.ndarray, out_path: Path,
                         thickness: int = 3):
    """원본 fundus 위에 예측 disc/cup 경계선을 테두리로 그려 저장.

    label_map(224 해상도)을 원본 이미지 크기로 키운 뒤 윤곽선을 올린다.
      disc(>=1) 테두리 = 초록,  cup(==2) 테두리 = 빨강
    """
    orig = Image.open(img_path).convert("RGB")
    W, H = orig.size
    rgb = np.asarray(orig).copy()

    # 예측 라벨맵을 원본 해상도로 (최근접) 업스케일
    lab = np.asarray(
        Image.fromarray(label_map.astype(np.uint8)).resize((W, H), Image.NEAREST)
    )

    disc = lab >= CFG.data.label_disc_rim     # 1∪2
    cup = lab >= CFG.data.label_cup           # 2
    # 두께는 이미지 크기에 비례(작은 크롭/큰 원본 모두 보기 좋게)
    t = max(2, round(thickness * max(W, H) / 224))

    rgb[_boundary(disc, t)] = [0, 220, 0]     # disc 경계 = 초록
    rgb[_boundary(cup, t)] = [255, 30, 30]    # cup 경계 = 빨강
    Image.fromarray(rgb).save(out_path)


def run_single(image: str):
    device = torch.device(CFG.runtime.device)
    model = load_model(device)
    img_path = Path(image)
    pred = predict_label(model, _preprocess(img_path, device))
    cdr = compute_cdr(pred)
    CFG.paths.ensure_dirs()
    # 원본 fundus 위에 예측 경계선을 테두리로 그려 저장
    out = CFG.paths.pred_dir / f"{img_path.stem}_contour.png"
    save_contour_overlay(img_path, pred, out)
    print(f"\n이미지: {img_path.name}")
    print(f"예측 C/D ratio ({CFG.infer.cdr_kind}) = {cdr:.4f}")
    print(f"경계선 오버레이 저장: {out}")


def run_split(split: str, save_vis: bool = False):
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
    for batch in loader:
        img = batch["image"].to(device)
        gt = batch["mask"][0].numpy().astype(np.int64)
        stem = batch["stem"][0]

        pred = predict_label(model, img)
        pc = compute_cdr(pred)
        gc = compute_cdr(gt)
        rows.append({"image": stem, "pred_cdr": pc, "gt_cdr": gc})
        if np.isfinite(pc) and np.isfinite(gc):
            pred_cdrs.append(pc)
            gt_cdrs.append(gc)

        if save_vis:
            save_contour_overlay(img_dir / f"{stem}.jpg", pred,
                                 vis_dir / f"{stem}_contour.png")

    csv_path = CFG.paths.pred_dir / f"cdr_{split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "pred_cdr", "gt_cdr"])
        w.writeheader()
        w.writerows(rows)

    pred_cdrs, gt_cdrs = np.array(pred_cdrs), np.array(gt_cdrs)
    mae = np.mean(np.abs(pred_cdrs - gt_cdrs))
    corr = np.corrcoef(pred_cdrs, gt_cdrs)[0, 1] if len(pred_cdrs) > 1 else float("nan")
    print(f"\n[{split}] n={len(rows)}  C/D ratio({CFG.infer.cdr_kind})")
    print(f"  MAE (pred vs GT-mask) = {mae:.4f}")
    print(f"  Pearson r            = {corr:.4f}")
    print(f"  결과 CSV: {csv_path}")
    if save_vis:
        print(f"  경계선 오버레이: {vis_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"],
                    help="해당 split 전체에 대해 C/D ratio 추론")
    ap.add_argument("--image", help="단일 이미지 경로 (전체 fundus 또는 크롭)")
    ap.add_argument("--save-vis", action="store_true",
                    help="split 전체에 대해 경계선 오버레이 이미지도 저장")
    args = ap.parse_args()

    if args.image:
        run_single(args.image)
    elif args.split:
        run_split(args.split, save_vis=args.save_vis)
    else:
        ap.error("--split 또는 --image 중 하나는 지정해야 합니다.")


if __name__ == "__main__":
    main()
