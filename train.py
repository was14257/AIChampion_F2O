"""
RETFound 인코더 + 디코더로 REFUGE disc/cup 세그멘테이션 학습.

실행:
    python train.py

설정은 전부 local_config.py(CFG) 에서 읽는다.
GPU 가 있으면 자동으로 CUDA 를 쓴다 (CFG.runtime.device).
검증 mean Dice 가 가장 좋은 모델을 outputs/checkpoints/best.pth 로 저장.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from local_config import CFG
from retfound_seg.dataset import build_loader
from retfound_seg.metrics import DiceCELoss, disc_cup_dice
from retfound_seg.model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda_factory(warmup, total):
    """워밍업 후 코사인 감쇠."""
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        prog = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return fn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    disc_sum, cup_sum, n = 0.0, 0.0, 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(img)
        d, c = disc_cup_dice(logits, mask)
        disc_sum += d.item()
        cup_sum += c.item()
        n += 1
    return disc_sum / n, cup_sum / n


def main():
    tc, rc = CFG.train, CFG.runtime
    device = torch.device(rc.device)
    set_seed(tc.seed)
    CFG.paths.ensure_dirs()

    print(f"device = {device}")
    train_loader = build_loader("train", tc.batch_size, shuffle=True,
                                train_aug=True)
    val_loader = build_loader("val", tc.batch_size, shuffle=False)
    print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    model = build_model(load_weights=True).to(device)

    enc_params, dec_params = model.param_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": tc.lr_encoder},
            {"params": dec_params, "lr": tc.lr_decoder},
        ],
        weight_decay=tc.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(tc.warmup_epochs, tc.epochs))
    criterion = DiceCELoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=tc.use_amp and device.type == "cuda")

    best_metric = -1.0
    for epoch in range(tc.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=tc.use_amp and device.type == "cuda"):
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            if tc.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        disc_dice, cup_dice = evaluate(model, val_loader, device)
        mean_dice = 0.5 * (disc_dice + cup_dice)
        dt = time.time() - t0
        print(f"[{epoch+1:03d}/{tc.epochs}] loss={running/len(train_loader):.4f} "
              f"val_disc={disc_dice:.4f} val_cup={cup_dice:.4f} "
              f"mean={mean_dice:.4f}  ({dt:.1f}s)")

        if mean_dice > best_metric:
            best_metric = mean_dice
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "mean_dice": mean_dice,
                "disc_dice": disc_dice,
                "cup_dice": cup_dice,
            }
            torch.save(ckpt, CFG.paths.ckpt_dir / "best.pth")
            print(f"    -> best 저장 (mean_dice={mean_dice:.4f})")
        if not tc.save_best_only:
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       CFG.paths.ckpt_dir / f"epoch_{epoch+1:03d}.pth")

    print(f"\n학습 완료. best mean_dice = {best_metric:.4f}")
    print(f"체크포인트: {CFG.paths.ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
