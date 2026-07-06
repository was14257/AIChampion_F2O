"""
사용 예시:

  # U-Net baseline 학습
  python train.py --model unet --data_root /path/to/REFUGE2 \
      --epochs 50 --batch_size 8 --output_dir runs/unet

  # RETFound 기반 segmentation 학습 (encoder frozen, decoder만 학습)
  python train.py --model retfound --data_root /path/to/REFUGE2 \
      --retfound_ckpt /path/to/RETFound_cfp_weights.pth \
      --epochs 50 --batch_size 8 --output_dir runs/retfound_frozen

  # RETFound encoder까지 미세조정 (더 낮은 lr 권장)
  python train.py --model retfound --data_root /path/to/REFUGE2 \
      --retfound_ckpt /path/to/RETFound_cfp_weights.pth --no_freeze_encoder \
      --lr 1e-4 --epochs 50 --output_dir runs/retfound_finetuned
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.refuge_dataset import RefugeSegDataset
from models.segmentation_models import build_model
from utils.losses import DiceCELoss, compute_dice_scores


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # 맥북 Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_od, total_oc, n_batches = 0.0, 0.0, 0.0, 0
    torch.set_grad_enabled(is_train)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores = compute_dice_scores(logits, masks)
        total_loss += loss.item()
        total_od += scores["od"]
        total_oc += scores["oc"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "dice_od": total_od / n_batches,
        "dice_oc": total_oc / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["retfound", "unet"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--retfound_ckpt", default=None)
    parser.add_argument("--no_freeze_encoder", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default="runs/exp")
    args = parser.parse_args()

    if args.model == "retfound" and args.retfound_ckpt is None:
        parser.error("--model retfound 는 --retfound_ckpt 가 필요해 (RETFound 사전학습 가중치 경로)")

    device = get_device()
    print(f"[device] {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RefugeSegDataset(args.data_root, split="train", img_size=args.img_size, augment=True)
    val_ds = RefugeSegDataset(args.data_root, split="val", img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    model = build_model(
        args.model,
        img_size=args.img_size,
        retfound_ckpt=args.retfound_ckpt,
        freeze_encoder=not args.no_freeze_encoder,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model} total_params={n_params:,} trainable_params={n_trainable:,}")

    criterion = DiceCELoss(num_classes=3)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = out_dir / "log.csv"
    best_mean_dice = -1.0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice_od", "val_dice_oc", "val_dice_mean", "sec"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
            val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None)
            scheduler.step()

            mean_dice = (val_stats["dice_od"] + val_stats["dice_oc"]) / 2
            elapsed = time.time() - t0
            print(
                f"[epoch {epoch:03d}] train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_dice_OD={val_stats['dice_od']:.4f} val_dice_OC={val_stats['dice_oc']:.4f} "
                f"({elapsed:.1f}s)"
            )
            writer.writerow(
                [epoch, train_stats["loss"], val_stats["loss"], val_stats["dice_od"], val_stats["dice_oc"], mean_dice, round(elapsed, 1)]
            )
            f.flush()

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                torch.save(
                    {"model_name": args.model, "state_dict": model.state_dict(), "epoch": epoch, "val_dice_mean": mean_dice},
                    out_dir / "best.pt",
                )

    print(f"[done] best val mean dice = {best_mean_dice:.4f} -> {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
