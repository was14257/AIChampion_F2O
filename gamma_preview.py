from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from local_config import PREVIEW_N, PREVIEW_CENTER

_CANDIDATES = [
    Path("D:/GAMMA"),
    Path("/d/GAMMA"),
    Path("/home/tta/data/GAMMA"),
]
GAMMA_ROOT = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
MM = "grading/Glaucoma_grading/{split}/multi-modality_images"


def list_cases(root: Path):
    pairs = []
    for split in ("training", "testing"):
        mm_dir = root / MM.format(split=split)
        if not mm_dir.exists():
            continue
        for case_dir in sorted(mm_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            cid = case_dir.name
            fundus = case_dir / f"{cid}.jpg"
            volume = case_dir / cid
            if fundus.exists() and volume.is_dir():
                pairs.append((cid, fundus, volume))
    return pairs


def central_bscan(volume_dir: Path, center: int = 128) -> Path | None:
    p = volume_dir / f"{center}_image.jpg"
    if p.exists():
        return p
    slices = sorted(volume_dir.glob("*_image.jpg"),
                    key=lambda x: int(x.stem.split("_")[0]))
    return slices[len(slices) // 2] if slices else None


def main():
    print(f"GAMMA_ROOT = {GAMMA_ROOT}")
    cases = list_cases(GAMMA_ROOT)
    print(f"Paired cases: {len(cases)}")
    if not cases:
        print("No cases found. Check the path.")
        return

    n_slices = len(list(cases[0][2].glob("*_image.jpg")))
    print(f"B-scan slices per case: {n_slices} (center index={PREVIEW_CENTER})")

    n = min(PREVIEW_N, len(cases))
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        cid, fundus_p, vol = cases[i]
        bscan_p = central_bscan(vol, PREVIEW_CENTER)
        fundus = Image.open(fundus_p).convert("RGB")
        bscan = Image.open(bscan_p).convert("L")

        axes[i, 0].imshow(fundus)
        axes[i, 0].set_title(f"{cid}  fundus {fundus.size}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(bscan, cmap="gray")
        axes[i, 1].set_title(f"central B-scan {bscan.size}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    out = Path("gamma_pair_preview.png")
    plt.savefig(out, dpi=90)
    print(f"Preview saved: {out.resolve()}")


if __name__ == "__main__":
    main()
