"""
REFUGE2 스타일 OD/OC segmentation 데이터셋.

기대하는 폴더 구조 (train/val/test 각각 동일 패턴):

    data_root/
        images/
            train/  xxx.jpg ...
            val/    xxx.jpg ...
            test/   xxx.jpg ...
        masks/
            train/  xxx.bmp ...   (또는 png)
            val/    ...
            test/   ...

마스크는 REFUGE 관례대로 그레이스케일 3-level:
    0   (검정)  = optic cup (OC)
    128 (회색)  = optic disc 영역이지만 cup 아님 (OD-ring)
    255 (흰색)  = background

만약 GRAPE나 다른 데이터셋 마스크 값이 다르면 MASK_VALUE_MAP만 바꿔주면 돼.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 그레이스케일 픽셀값 -> 클래스 인덱스 (0=bg, 1=OD-ring, 2=OC)
MASK_VALUE_MAP = {255: 0, 128: 1, 0: 2}


def _map_mask(mask_arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask_arr, dtype=np.int64)
    for val, cls in MASK_VALUE_MAP.items():
        out[mask_arr == val] = cls
    return out


class RefugeSegDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
        image_ext=(".jpg", ".jpeg", ".png"),
        mask_ext=(".bmp", ".png"),
    ):
        self.img_dir = Path(data_root) / "images" / split
        self.mask_dir = Path(data_root) / "masks" / split
        self.img_size = img_size
        self.augment = augment and split == "train"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"이미지 폴더가 없어: {self.img_dir}")

        stems = sorted(
            p.stem for p in self.img_dir.iterdir() if p.suffix.lower() in image_ext
        )
        self.samples = []
        for stem in stems:
            img_path = next(
                (self.img_dir / f"{stem}{e}" for e in image_ext if (self.img_dir / f"{stem}{e}").exists()),
                None,
            )
            mask_path = next(
                (self.mask_dir / f"{stem}{e}" for e in mask_ext if (self.mask_dir / f"{stem}{e}").exists()),
                None,
            )
            if img_path is not None and mask_path is not None:
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"{self.img_dir} / {self.mask_dir} 에서 짝이 맞는 이미지-마스크를 못 찾았어. "
                f"파일명(stem)이 이미지-마스크 간에 동일한지 확인해줘."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        if self.augment:
            if np.random.rand() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, IMAGENET_MEAN, IMAGENET_STD)

        mask_arr = _map_mask(np.array(mask))
        mask_t = torch.from_numpy(mask_arr).long()

        return image_t, mask_t
