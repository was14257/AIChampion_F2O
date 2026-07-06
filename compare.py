"""
학습된 두 체크포인트(RETFound / U-Net)를 같은 test split에서 비교.

사용 예시:
  python compare.py --data_root /path/to/REFUGE2 \
      --unet_ckpt runs/unet/best.pt \
      --retfound_ckpt runs/retfound_frozen/best.pt \
      --retfound_weights /path/to/RETFound_cfp_weights.pth
"""
import argparse

import torch
from torch.utils.data import DataLoader

from data.refuge_dataset import RefugeSegDataset
from models.segmentation_models import UNetBaseline, RETFoundSegmenter
from utils.losses import compute_dice_scores


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_od, total_oc, n = 0.0, 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        scores = compute_dice_scores(logits, masks)
        total_od += scores["od"]
        total_oc += scores["oc"]
        n += 1
    return total_od / n, total_oc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--unet_ckpt", required=True)
    parser.add_argument("--retfound_ckpt", required=True)
    parser.add_argument("--retfound_weights", required=True, help="RETFoundEncoder 아키텍처 구성용 원본 사전학습 가중치 경로")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    test_ds = RefugeSegDataset(args.data_root, split="test", img_size=args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"[data] test={len(test_ds)}")

    results = {}

    # U-Net
    unet = UNetBaseline().to(device)
    ckpt = torch.load(args.unet_ckpt, map_location=device)
    unet.load_state_dict(ckpt["state_dict"])
    od, oc = evaluate(unet, test_loader, device)
    results["U-Net (from scratch)"] = (od, oc, sum(p.numel() for p in unet.parameters()))

    # RETFound
    retfound = RETFoundSegmenter(img_size=args.img_size, checkpoint_path=args.retfound_weights, freeze_encoder=True).to(device)
    ckpt = torch.load(args.retfound_ckpt, map_location=device)
    retfound.load_state_dict(ckpt["state_dict"])
    od, oc = evaluate(retfound, test_loader, device)
    results["RETFound + Segmenter decoder"] = (od, oc, sum(p.numel() for p in retfound.parameters()))

    print("\n=== 비교 결과 (Test set) ===")
    print(f"{'Model':<32}{'Dice OD':>10}{'Dice OC':>10}{'Mean':>10}{'#Params':>14}")
    for name, (od, oc, n_params) in results.items():
        mean = (od + oc) / 2
        print(f"{name:<32}{od:>10.4f}{oc:>10.4f}{mean:>10.4f}{n_params:>14,}")


if __name__ == "__main__":
    main()
