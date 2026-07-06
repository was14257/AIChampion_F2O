"""
REFUGE2 / GAMMA / GRAPE 등 서로 다른 원본 배포 구조를 train.py가 기대하는
표준 구조로 정규화하는 변환 스크립트.

표준 구조 (README 참고):
    out_root/
        images/{train,val,test}/xxx.jpg
        masks/{train,val,test}/xxx.png   (그레이스케일, 255=bg/128=OD-ring/0=OC)

원본 데이터셋마다 폴더명·마스크 픽셀값 관례가 달라서, 이 스크립트는
"이미지 폴더 재귀 스캔 + 마스크 폴더 재귀 스캔 -> 파일명(stem) 기준 매칭"
방식으로 동작하고, 마스크 픽셀값 매핑은 샘플을 보고 자동 추정하되
--mask_map으로 직접 override 가능하게 해뒀어.

사용 예시 (REFUGE2, GAMMA처럼 train/val/test가 원래 폴더로 나뉜 배포본):
    # 공식 배포 zip을 풀면 보통 train/val/test가 각각 별도 폴더로 옴 ->
    # 3번 나눠서 호출 (매번 --split 지정)
    python prepare_dataset.py \
        --images_dir /path/to/REFUGE-Training400 \
        --masks_dir /path/to/Annotation-Training400 \
        --out_root ./REFUGE2 --split train

    python prepare_dataset.py \
        --images_dir /path/to/REFUGE-Validation400 \
        --masks_dir /path/to/REFUGE-Validation400-GT \
        --out_root ./REFUGE2 --split val

사용 예시 (GRAPE처럼 공식 train/val/test 구분이 없는 단일 코호트):
    # CFPs = 원본 이미지, "Segmentation masks" = 실제 정답 마스크
    # (grape_annotated/ 처럼 초록/빨강 윤곽선이 이미지 위에 합성된 "Annotated Images"는
    #  시각화용이라 정답 마스크로 못 씀 - 반드시 별도 배포되는 마스크 파일을 써야 해)
    python prepare_dataset.py \
        --images_dir /path/to/GRAPE_CFPs \
        --masks_dir /path/to/GRAPE_Segmentation_masks \
        --out_root ./GRAPE --split_ratios 0.7,0.15,0.15 --seed 42

--dry_run을 붙이면 실제로 복사/링크하지 않고 몇 장이 매칭되는지,
마스크 값이 어떻게 해석됐는지만 보여줘 (본 실행 전에 확인 용도).
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
# 마스크 파일명에 흔히 붙는 접미사 (이미지 stem과 직접 매칭 안 될 때 벗겨보고 재시도)
MASK_SUFFIXES = ("_mask", "_masks", "_gt", "_seg", "_segmentation", "_label", "_labels", "_disc_cup", "-mask", "-gt")


def index_by_stem(root: Path, exts):
    """root 아래를 재귀로 뒤져서 stem -> path 매핑을 만든다. stem 중복이면 첫 번째만 쓰고 경고."""
    found = {}
    dupes = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in found:
                dupes += 1
                continue
            found[p.stem] = p
    if dupes:
        print(f"[warn] {root} 아래에서 stem 중복 {dupes}건 발견 -> 먼저 찾은 파일만 사용")
    return found


def match_by_stem(image_stems, mask_index):
    """이미지 stem 집합을 마스크 인덱스와 매칭. 정확히 안 맞으면 흔한 접미사를 벗겨서 재시도."""
    direct = set(image_stems) & set(mask_index)
    remaining = set(image_stems) - direct
    if not remaining:
        return {s: mask_index[s] for s in direct}

    # 접미사 제거한 stem -> 원본 마스크 경로
    stripped_index = {}
    for stem, path in mask_index.items():
        for suf in MASK_SUFFIXES:
            if stem.endswith(suf):
                stripped_index.setdefault(stem[: -len(suf)], path)
                break

    matched = {s: mask_index[s] for s in direct}
    recovered = 0
    for s in remaining:
        if s in stripped_index:
            matched[s] = stripped_index[s]
            recovered += 1
    if recovered:
        print(f"[match] 마스크 파일명 접미사({'/'.join(MASK_SUFFIXES)}) 제거 후 추가 매칭 {recovered}건")
    return matched


def parse_mask_map(spec: str):
    mapping = {}
    for part in spec.split(","):
        val, cls = part.split(":")
        mapping[int(val)] = int(cls)
    return mapping


def detect_mask_map(mask_paths, sample_size=20):
    """마스크 샘플들의 unique 픽셀값을 보고 REFUGE(0/128/255) 또는 (0/1/2) 관례인지 추정."""
    values = set()
    for p in mask_paths[:sample_size]:
        arr = np.array(Image.open(p).convert("L"))
        values.update(np.unique(arr).tolist())

    if values <= {0, 128, 255}:
        mapping = {255: 0, 128: 1, 0: 2}
        print(f"[detect] 마스크 픽셀값 {sorted(values)} -> REFUGE 관례로 판단 (255=bg,128=OD-ring,0=OC)")
    elif values <= {0, 1, 2}:
        mapping = {0: 0, 1: 1, 2: 2}
        print(f"[detect] 마스크 픽셀값 {sorted(values)} -> 이미 클래스 인덱스(0/1/2)로 판단")
    else:
        raise SystemExit(
            f"[error] 마스크 픽셀값을 자동으로 해석 못 함: {sorted(values)}\n"
            f"  --mask_map \"val:cls,...\" 로 직접 지정해줘 (cls: 0=bg,1=OD-ring,2=OC).\n"
            f"  값이 2개뿐이면(예: {{0,255}}) OD만 있는 마스크일 수 있음 -> "
            f"--od_mask_dir/--oc_mask_dir로 OD/OC를 분리해서 넣는 방법도 있어."
        )
    return mapping


def remap_mask(mask_path: Path, mapping: dict) -> Image.Image:
    arr = np.array(Image.open(mask_path).convert("L"))
    out = np.zeros_like(arr, dtype=np.uint8)
    for val, cls in mapping.items():
        cls_to_pixel = {0: 255, 1: 128, 2: 0}
        out[arr == val] = cls_to_pixel[cls]
    return Image.fromarray(out, mode="L")


def combine_od_oc(od_path: Path, oc_path: Path) -> Image.Image:
    od = np.array(Image.open(od_path).convert("L")) > 0
    oc = np.array(Image.open(oc_path).convert("L")) > 0
    out = np.full(od.shape, 255, dtype=np.uint8)  # background
    out[od] = 128  # OD-ring
    out[oc] = 0  # OC (cup wins over ring if overlapping)
    return Image.fromarray(out, mode="L")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True, help="원본 이미지 폴더 (재귀 탐색)")
    ap.add_argument("--masks_dir", default=None, help="원본 마스크 폴더 (재귀 탐색, 3-level 단일 마스크 파일 방식)")
    ap.add_argument("--od_mask_dir", default=None, help="OD만 있는 바이너리 마스크 폴더 (masks_dir 대신 od+oc 조합 방식일 때)")
    ap.add_argument("--oc_mask_dir", default=None, help="OC만 있는 바이너리 마스크 폴더")
    ap.add_argument("--out_root", required=True, help="정규화된 결과를 쓸 폴더 (train.py --data_root에 그대로 넣으면 됨)")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None, help="지정하면 이 split에만 추가 (공식 train/val/test가 이미 나뉜 배포본용)")
    ap.add_argument("--split_ratios", default="0.7,0.15,0.15", help="--split 생략 시 전체 풀을 이 비율로 무작위 분할 (train,val,test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask_map", default=None, help='마스크 픽셀값->클래스 매핑 강제 지정, 예: "255:0,128:1,0:2"')
    ap.add_argument("--copy", action="store_true", help="심볼릭 링크 대신 실제 복사 (기본은 심볼릭 링크)")
    ap.add_argument("--dry_run", action="store_true", help="실제로 쓰지 않고 매칭 결과만 출력")
    args = ap.parse_args()

    if not args.masks_dir and not (args.od_mask_dir and args.oc_mask_dir):
        ap.error("--masks_dir 또는 (--od_mask_dir + --oc_mask_dir) 중 하나는 필요해")

    images_dir = Path(args.images_dir)
    out_root = Path(args.out_root)
    images = index_by_stem(images_dir, IMAGE_EXTS)
    print(f"[scan] 이미지 {len(images)}장 발견 ({images_dir})")

    use_od_oc = bool(args.od_mask_dir and args.oc_mask_dir)
    if use_od_oc:
        od_masks = index_by_stem(Path(args.od_mask_dir), MASK_EXTS)
        oc_masks = index_by_stem(Path(args.oc_mask_dir), MASK_EXTS)
        od_matched = match_by_stem(images.keys(), od_masks)
        oc_matched = match_by_stem(images.keys(), oc_masks)
        stems = sorted(set(od_matched) & set(oc_matched))
        od_masks, oc_masks = od_matched, oc_matched
        print(f"[scan] OD 마스크 {len(od_masks)}장, OC 마스크 {len(oc_masks)}장 -> 3자 매칭 {len(stems)}장")
        mapping = None
    else:
        raw_masks = index_by_stem(Path(args.masks_dir), MASK_EXTS)
        masks = match_by_stem(images.keys(), raw_masks)
        stems = sorted(masks)
        print(f"[scan] 마스크 {len(raw_masks)}장 -> 이미지와 매칭 {len(stems)}장")
        if not stems:
            raise SystemExit("[error] 이미지-마스크 짝을 하나도 못 찾음. 파일명(stem)이 서로 같은지 확인해줘.")
        mapping = parse_mask_map(args.mask_map) if args.mask_map else detect_mask_map([masks[s] for s in stems])

    missing = len(images) - len(stems) if not use_od_oc else len(images) - len(stems)
    if missing > 0:
        print(f"[warn] 이미지는 있는데 마스크가 없어서 제외된 stem {missing}개")
    if not stems:
        raise SystemExit("[error] 매칭되는 샘플이 하나도 없어.")

    # split 배정
    if args.split:
        assignment = {s: args.split for s in stems}
    else:
        ratios = [float(x) for x in args.split_ratios.split(",")]
        assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "--split_ratios 는 세 개 합이 1.0이어야 함"
        rng = random.Random(args.seed)
        shuffled = stems[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        assignment = {}
        for s in shuffled[:n_train]:
            assignment[s] = "train"
        for s in shuffled[n_train:n_train + n_val]:
            assignment[s] = "val"
        for s in shuffled[n_train + n_val:]:
            assignment[s] = "test"
        print(f"[split] train={n_train} val={n_val} test={n - n_train - n_val} (seed={args.seed})")

    if args.dry_run:
        print("[dry_run] 실제 파일은 쓰지 않음. 위 매칭/분할 결과만 확인하고 끝냄.")
        return

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "masks" / split).mkdir(parents=True, exist_ok=True)

    n_written = 0
    for stem in stems:
        split = assignment[stem]
        img_src = images[stem]
        img_dst = out_root / "images" / split / img_src.name
        if args.copy:
            shutil.copy2(img_src, img_dst)
        else:
            if img_dst.exists() or img_dst.is_symlink():
                img_dst.unlink()
            img_dst.symlink_to(img_src.resolve())

        mask_dst = out_root / "masks" / split / f"{stem}.png"
        if use_od_oc:
            mask_img = combine_od_oc(od_masks[stem], oc_masks[stem])
        else:
            mask_img = remap_mask(masks[stem], mapping)
        mask_img.save(mask_dst)
        n_written += 1

    print(f"[done] {n_written}장 -> {out_root} (images/masks 각각 train/val/test)")
    print(f"이제 이 폴더를 그대로 --data_root 로 넘기면 돼: python train.py --model unet --data_root {out_root} ...")


if __name__ == "__main__":
    main()
