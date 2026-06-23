"""
세그멘테이션 마스크로부터 C/D ratio(cup-to-disc ratio) 계산.

마스크 라벨: 0=배경, 1=disc rim, 2=cup
  disc = (라벨>=1),  cup = (라벨==2)

제공 함수
  - vertical_cdr : 임상 표준(수직 컵직경 / 수직 디스크직경)
  - area_cdr     : 면적비(컵면적 / 디스크면적)
  - compute_cdr  : InferConfig.cdr_kind 에 따라 선택
"""
from __future__ import annotations

import numpy as np

from local_config import CFG

# scipy 가 있으면 가장 큰 연결요소만 남기는 후처리를 쓴다 (없어도 동작).
try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _keep_largest(binary: np.ndarray) -> np.ndarray:
    """가장 큰 연결요소만 남긴다 (잡음 제거)."""
    if not (_HAS_SCIPY and CFG.infer.keep_largest_cc) or binary.sum() == 0:
        return binary
    labeled, n = _ndi.label(binary)
    if n <= 1:
        return binary
    sizes = _ndi.sum(binary, labeled, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labeled == keep


def _vertical_diameter(binary: np.ndarray) -> float:
    """이진 마스크의 수직 직경 = 어떤 열에서든 가장 긴 세로 연속/총 길이.

    임상 정의에 가깝게, 각 열(column)별 전경 픽셀 수의 최댓값을 직경으로 본다.
    """
    if binary.sum() == 0:
        return 0.0
    col_heights = binary.sum(axis=0)      # 열마다 세로 픽셀 수
    return float(col_heights.max())


def masks_from_label(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """라벨맵(0/1/2) -> (disc 이진, cup 이진)."""
    disc = _keep_largest(label_map >= CFG.data.label_disc_rim)
    cup = _keep_largest(label_map >= CFG.data.label_cup)
    return disc, cup


def postprocess_label(label_map: np.ndarray) -> np.ndarray:
    """예측 라벨맵 정리: 가장 큰 disc 덩어리 1개만 남기고 오탐 제거.

    - 전체 fundus 추론 시 밝은 반사광에 생기는 가짜 disc/cup 을 없앤다.
    - cup 은 살아남은 disc 영역 안의 것만 유지한다.
    반환: 정리된 라벨맵(0/1/2).
    """
    disc = _keep_largest(label_map >= CFG.data.label_disc_rim)  # 최대 덩어리
    cup = (label_map >= CFG.data.label_cup) & disc              # disc 안의 cup만
    cup = _keep_largest(cup)

    out = np.zeros_like(label_map)
    out[disc] = CFG.data.label_disc_rim
    out[cup] = CFG.data.label_cup
    return out


def vertical_cdr(label_map: np.ndarray) -> float:
    disc, cup = masks_from_label(label_map)
    d = _vertical_diameter(disc)
    if d == 0:
        return float("nan")
    return _vertical_diameter(cup) / d


def area_cdr(label_map: np.ndarray) -> float:
    disc, cup = masks_from_label(label_map)
    d = float(disc.sum())
    if d == 0:
        return float("nan")
    return float(cup.sum()) / d


def compute_cdr(label_map: np.ndarray, kind: str | None = None) -> float:
    kind = kind or CFG.infer.cdr_kind
    if kind == "vertical":
        return vertical_cdr(label_map)
    if kind == "area":
        return area_cdr(label_map)
    raise ValueError(f"cdr_kind 는 'vertical' 또는 'area': {kind}")
