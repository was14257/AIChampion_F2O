"""
REFUGE 시신경유두/함몰부 세그멘테이션 데이터셋.

ROI 크롭본(Images_Cropped / Masks_Cropped)을 사용한다.
마스크 라벨: 0=배경, 1=disc rim, 2=cup  (local_config.DataConfig 참고)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from local_config import CFG


# 크롭 이미지 확장자: jpg, 마스크: png (REFUGE 규약)
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
    """REFUGE 한 split(train/val/test)의 세그멘테이션 데이터셋."""

    def __init__(self, split: str, train_aug: bool = False):
        self.split = split
        self.train_aug = train_aug
        self.cfg = CFG.data

        p = CFG.paths
        dirs = {
            "train": (p.refuge_train_img, p.refuge_train_mask),
            "val": (p.refuge_val_img, p.refuge_val_mask),
            "test": (p.refuge_test_img, p.refuge_test_mask),
        }
        if split not in dirs:
            raise ValueError(f"split 은 train/val/test 중 하나여야 함: {split}")
        img_dir, mask_dir = dirs[split]
        self.pairs = _pair_files(img_dir, mask_dir)

        self.mean = torch.tensor(self.cfg.mean).view(3, 1, 1)
        self.std = torch.tensor(self.cfg.std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    # --- 변환 -------------------------------------------------------------- #
    def _load(self, img_path: Path, mask_path: Path):
        size = self.cfg.img_size
        img = Image.open(img_path).convert("RGB").resize(
            (size, size), Image.BILINEAR
        )
        # 마스크는 라벨값이므로 최근접 보간으로 리사이즈
        mask = Image.open(mask_path).convert("L").resize(
            (size, size), Image.NEAREST
        )
        img = np.asarray(img, dtype=np.float32) / 255.0       # HxWx3
        mask = np.asarray(mask, dtype=np.int64)               # HxW (0/1/2)
        return img, mask

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        """가벼운 기하/광학 증강 (학습 시에만)."""
        # 좌우/상하 반전
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1, :], mask[:, ::-1]
        if np.random.rand() < 0.5:
            img, mask = img[::-1, :, :], mask[::-1, :]
        # 90도 단위 회전
        k = np.random.randint(4)
        if k:
            img = np.rot90(img, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
        # 밝기/대비 지터
        if np.random.rand() < 0.5:
            gain = 1.0 + (np.random.rand() - 0.5) * 0.4   # 0.8~1.2
            bias = (np.random.rand() - 0.5) * 0.1
            img = np.clip(img * gain + bias, 0.0, 1.0)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img, mask = self._load(img_path, mask_path)
        if self.train_aug:
            img, mask = self._augment(img, mask)

        img_t = torch.from_numpy(img).permute(2, 0, 1)        # 3xHxW
        img_t = (img_t - self.mean) / self.std
        mask_t = torch.from_numpy(mask)                       # HxW

        return {
            "image": img_t,
            "mask": mask_t,
            "stem": img_path.stem,
        }


def build_loader(split: str, batch_size: int, shuffle: bool,
                 train_aug: bool = False) -> DataLoader:
    ds = RefugeSegDataset(split, train_aug=train_aug)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.data.pin_memory,
        drop_last=shuffle,
    )
