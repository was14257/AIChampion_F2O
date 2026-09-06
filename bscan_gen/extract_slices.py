import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG

ROOT = CFG.paths.gamma_grading


def extract_case(mhd_path: Path, out_dir: Path, overwrite=False):
    """Extract all B-scan slices from an .mhd volume and save each as a jpg."""
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mhd_path)))  # (Z, H, W) uint8
    out_dir.mkdir(parents=True, exist_ok=True)
    for si in range(vol.shape[0]):
        fp = out_dir / f"{si}_image.jpg"
        if overwrite or not fp.exists():
            Image.fromarray(vol[si]).save(fp, quality=95)
    return vol.shape[0]


def main():
    """Extract slices for every case in training/testing that hasn't been extracted yet."""
    n_cases = 0
    n_slices = 0
    for split in ("training", "testing"):
        base = ROOT / split / "multi-modality_images"
        if not base.exists():
            continue
        for cdir in sorted(base.iterdir()):
            if not cdir.is_dir():
                continue
            cid = cdir.name
            mhd = cdir / f"{cid}_Sequence.mhd"
            if not mhd.exists():
                continue
            out_dir = cdir / cid
            existing = len(list(out_dir.glob("*_image.jpg"))) if out_dir.exists() else 0
            if existing >= 256:
                continue
            n = extract_case(mhd, out_dir)
            n_cases += 1
            n_slices += n
            print(f"{split}/{cid}: {n} slices")
    print(f"done: {n_cases} cases, {n_slices} slices")


if __name__ == "__main__":
    main()
