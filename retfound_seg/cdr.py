import numpy as np

from config import CFG

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _keep_largest(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    if not (_HAS_SCIPY and CFG.infer.keep_largest_cc) or binary.sum() == 0:
        return binary
    labeled, n = _ndi.label(binary)
    if n <= 1:
        return binary
    sizes = _ndi.sum(binary, labeled, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labeled == keep


def _vertical_diameter(binary: np.ndarray) -> float:
    """Max column height (vertical extent) of a binary mask."""
    if binary.sum() == 0:
        return 0.0
    col_heights = binary.sum(axis=0)
    return float(col_heights.max())


def masks_from_label(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract disc and cup binary masks from a label map."""
    disc = _keep_largest(label_map >= CFG.data.label_disc_rim)
    cup = _keep_largest(label_map >= CFG.data.label_cup)
    return disc, cup


def postprocess_label(label_map: np.ndarray) -> np.ndarray:
    """Clean up a raw label map by keeping only the largest disc/cup components."""
    disc = _keep_largest(label_map >= CFG.data.label_disc_rim)
    cup = (label_map >= CFG.data.label_cup) & disc
    cup = _keep_largest(cup)

    out = np.zeros_like(label_map)
    out[disc] = CFG.data.label_disc_rim
    out[cup] = CFG.data.label_cup
    return out


def vertical_cdr(label_map: np.ndarray) -> float:
    """Vertical cup-to-disc ratio from a label map."""
    disc, cup = masks_from_label(label_map)
    d = _vertical_diameter(disc)
    if d == 0:
        return float("nan")
    return _vertical_diameter(cup) / d


def area_cdr(label_map: np.ndarray) -> float:
    """Area-based cup-to-disc ratio from a label map."""
    disc, cup = masks_from_label(label_map)
    d = float(disc.sum())
    if d == 0:
        return float("nan")
    return float(cup.sum()) / d


def compute_cdr(label_map: np.ndarray, kind: str | None = None) -> float:
    """Dispatch to vertical or area cup-to-disc ratio computation."""
    kind = kind or CFG.infer.cdr_kind
    if kind == "vertical":
        return vertical_cdr(label_map)
    if kind == "area":
        return area_cdr(label_map)
    raise ValueError(f"cdr_kind must be 'vertical' or 'area': {kind}")
