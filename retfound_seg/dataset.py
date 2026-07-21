from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import CFG


def _pair_files(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    if not pairs:
        raise FileNotFoundError(
            f"이미지/마스크 쌍을 찾지 못함: {img_dir} <-> {mask_dir}"
        )
    return pairs


class RefugeSegDataset(Dataset):
    def __init__(self, split: str, train_aug: bool = False,
                 merge_train_val: bool = False):
        self.split = split
        self.train_aug = train_aug
        self.cfg = CFG.data

        p = CFG.paths

        if merge_train_val and split in ('train', 'val'):
            all_pairs = (
                _pair_files(p.refuge_train_img, p.refuge_train_mask) +
                _pair_files(p.refuge_val_img,   p.refuge_val_mask)
            )
            all_pairs.sort(key=lambda x: x[0].name)
            if split == 'train':
                self.pairs = all_pairs[:600]
            else:
                self.pairs = all_pairs[600:]
        else:
            dirs = {
                "train": (p.refuge_train_img, p.refuge_train_mask),
                "val":   (p.refuge_val_img,   p.refuge_val_mask),
                "test":  (p.refuge_test_img,  p.refuge_test_mask),
            }
            if split not in dirs:
                raise ValueError(f"split 은 train/val/test 중 하나여야 함: {split}")
            img_dir, mask_dir = dirs[split]
            self.pairs = _pair_files(img_dir, mask_dir)

        size = self.cfg.img_size
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=180, border_mode=0, value=0, mask_value=0, p=0.8),
            A.RandomResizedCrop(
                size=(size, size), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.5
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
        ]) if train_aug else None

        self.mean = torch.tensor(self.cfg.mean).view(3, 1, 1)
        self.std = torch.tensor(self.cfg.std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, img_path: Path, mask_path: Path):
        size = self.cfg.img_size
        img = Image.open(img_path).convert("RGB").resize(
            (size, size), Image.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (size, size), Image.NEAREST
        )
        img  = np.asarray(img,  dtype=np.uint8)
        mask = np.asarray(mask, dtype=np.uint8)
        return img, mask

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img, mask = self._load(img_path, mask_path)

        if self.aug is not None:
            out  = self.aug(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        img_t  = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_t  = (img_t - self.mean) / self.std
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": img_t,
            "mask":  mask_t,
            "stem":  img_path.stem,
        }


def build_loader(split: str, batch_size: int, shuffle: bool,
                 train_aug: bool = False,
                 merge_train_val: bool = False) -> DataLoader:
    ds = RefugeSegDataset(split, train_aug=train_aug, merge_train_val=merge_train_val)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.data.pin_memory,
        drop_last=shuffle,
    )
