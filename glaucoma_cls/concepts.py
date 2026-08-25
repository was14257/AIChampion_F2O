"""Concept extraction: disc/cup mask geometric concepts (6) + fundus
embedding->OCT concepts (3).

Originally merged from cbm/concepts.py + cbm/oct_head.py.
- seg_concepts(): disc/cup segmentation mask -> 6 concepts (cdr/ovality/area etc.)
- fit_oct_linear(): whole+disc embedding -> linear regression for
  [mean_th, ilm_rough, fovea_curv]
  (called only from the training script scripts/fit_oct_linear.py; the
  server only loads oct_linear.pt)

SEG_CONCEPTS(6) + OCT_CONCEPTS(3) = ALL_CONCEPTS(9), fixed order.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

from retfound_seg.cdr import masks_from_label, compute_cdr

SEG_CONCEPTS = ["cdr", "horizontal_cdr", "disc_ovality", "cup_ovality",
                "disc_area", "rim_area"]
OCT_CONCEPTS = ["mean_th", "ilm_rough", "fovea_curv"]
ALL_CONCEPTS = SEG_CONCEPTS + OCT_CONCEPTS  # 9 total, fixed order

# v2 (2026-08-09): OCT_CONCEPTS(3, GAMMA 172-image hand-labeled regression)
# replaced by RNFL_CONCEPTS(5), a fundus->RNFL-thickness regression trained
# on GRAPE (244 real OCT RNFL measurements: Mean/S/N/I/T quadrants). GRAPE
# gives more data and covers a wider severity range (all-glaucoma cohort),
# and 5-fold CV inside GRAPE confirmed r=0.39~0.83 (weakest on N, the
# quadrant with the least inter-subject variance and least glaucoma
# sensitivity - see grape_train_rnfl.py). Order fixed as mean/I/S/N/T
# (ISNT rule order used clinically, mean first).
RNFL_CONCEPTS = ["rnfl_mean", "rnfl_I", "rnfl_S", "rnfl_N", "rnfl_T"]
ALL_CONCEPTS_V2 = SEG_CONCEPTS + RNFL_CONCEPTS  # 11 total, fixed order

# Display metadata: (label, unit, short description, normal range(lo,hi), risk direction).
# Normal range is computed from cbm_concepts.npz (REFUGE+ORIGA+G1020+GAMMA,
# n=2127, y=glaucoma label), as the IQR (25th-75th percentile) of the
# label=0 (normal) group (one-off scratchpad script, 2026-07-27). Risk
# direction confirmed empirically by comparing normal-group vs
# glaucoma-group medians - rim_area intuitively suggests "thinner = riskier"
# by name, but the measured result was the opposite (glaucoma-group median
# is larger, likely due to correlation with rising CDR and enlarging
# disc_area), so we reflect the measured value as-is.
# disc/cup area is a pixel count from the 512x512 seg mask, so it's for
# relative comparison rather than an absolute value.
CONCEPT_META = {
    "cdr":            ("Vertical C/D ratio", "",     "Cup/Disc vertical diameter ratio (0~1, higher=riskier)",
                        (0.379, 0.488), "high"),
    "horizontal_cdr": ("Horizontal C/D ratio", "",     "Cup/Disc horizontal diameter ratio (0~1)",
                        (0.429, 0.521), "high"),
    "disc_ovality":   ("Disc ovality", "x", "major/minor axis ratio (1=circle, higher=more distorted)",
                        (1.227, 1.399), "high"),
    "cup_ovality":    ("Cup ovality", "x",    "major/minor axis ratio (1=circle)",
                        (1.108, 1.336), "high"),
    "disc_area":      ("Optic disc area", "px",  "seg mask pixel count (based on 512x512)",
                        (4897, 6563), "high"),
    "rim_area":       ("Neuroretinal rim area", "px",  "disc-cup pixel count",
                        (3836, 5168), "high"),
    "mean_th":        ("Mean retinal thickness", "um",   "predicted value (thinner=riskier)",
                        (111.1, 118.0), "low"),
    "ilm_rough":      ("ILM surface roughness", "um",  "top retinal layer unevenness (predicted value)",
                        (4.13, 5.72), "low"),
    "fovea_curv":     ("Foveal curvature", "1/px",   "foveal concavity (predicted value, negative=concave)",
                        (-0.019, -0.013), "high"),
    # v2 RNFL concepts: normal range = IQR of label=0 group in
    # cbm_concepts_v2.npz (GAMMA_train+REFUGE+GRAPE, n=758). Predicted from
    # fundus via a GRAPE(n=244 real OCT RNFL)-trained regression, thinner=riskier.
    "rnfl_mean":      ("RNFL Mean thickness", "um", "peripapillary RNFL average thickness (predicted, thinner=riskier)",
                        (96.3, 111.5), "low"),
    "rnfl_I":         ("RNFL Inferior thickness", "um", "inferior quadrant RNFL thickness (predicted)",
                        (116.3, 137.2), "low"),
    "rnfl_S":         ("RNFL Superior thickness", "um", "superior quadrant RNFL thickness (predicted)",
                        (116.3, 136.3), "low"),
    "rnfl_N":         ("RNFL Nasal thickness", "um", "nasal quadrant RNFL thickness (predicted, least sensitive to glaucoma)",
                        (76.1, 84.5), "low"),
    "rnfl_T":         ("RNFL Temporal thickness", "um", "temporal quadrant RNFL thickness (predicted)",
                        (75.5, 90.4), "low"),
}


# --- disc/cup mask geometric concepts ---

def _ovality(binary: np.ndarray) -> float:
    """Major/minor axis ratio. Higher values mean the disc (or cup) is more distorted."""
    binary = binary.astype(np.uint8)
    if binary.sum() < 5:
        return float("nan")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float("nan")
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return float("nan")
    (_, _), (minor, major), _ = cv2.fitEllipse(largest)
    if minor == 0:
        return float("nan")
    return float(major / minor)


def _horizontal_diameter(binary: np.ndarray) -> float:
    if binary.sum() == 0:
        return 0.0
    row_widths = binary.sum(axis=1)
    return float(row_widths.max())


def _horizontal_cdr(disc: np.ndarray, cup: np.ndarray) -> float:
    d = _horizontal_diameter(disc)
    if d == 0:
        return float("nan")
    return _horizontal_diameter(cup) / d


def seg_concepts(label_map: np.ndarray, cdr_kind: str = "vertical") -> dict:
    """label_map: an (H,W) integer array from postprocess_label() (0=bg,1=disc/rim,2=cup)."""
    disc, cup = masks_from_label(label_map)
    disc_area = float(disc.sum())
    cup_area = float(cup.sum())
    return {
        "cdr": compute_cdr(label_map, kind=cdr_kind),
        "horizontal_cdr": _horizontal_cdr(disc, cup),
        "disc_ovality": _ovality(disc),
        "cup_ovality": _ovality(cup),
        "disc_area": disc_area,
        "rim_area": disc_area - cup_area,
    }


# --- fundus embedding -> OCT concept linear regression ---

def fit_oct_linear(X: np.ndarray, Y: np.ndarray, n_components: int = 8) -> nn.Linear:
    """X: (N, D) whole+disc embedding concat, Y: (N, len(OCT_CONCEPTS)) targets.
    Trains with sklearn PLS + StandardScaler, then freezes the resulting
    coefficients into an nn.Linear for return. Since PLS is a linear
    transform of the form y = (x - x_mean) @ ... + y_mean, it can be folded
    together with the scaler into a single affine transform. (Training-script
    only; sklearn is imported inside the function only.)
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pls = PLSRegression(n_components=n_components).fit(Xs, Y)

    coef = pls.coef_  # (n_targets, D) or (D, n_targets) — depends on sklearn version
    if coef.shape[0] != len(OCT_CONCEPTS):
        coef = coef.T
    intercept = np.atleast_1d(pls.intercept_)

    scale = scaler.scale_  # (D,)
    mean = scaler.mean_    # (D,)

    W = coef / scale[None, :]                      # (n_targets, D)
    b = intercept - (coef * (mean / scale)[None, :]).sum(axis=1)  # (n_targets,)

    linear = nn.Linear(X.shape[1], len(OCT_CONCEPTS))
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(W).float())
        linear.bias.copy_(torch.from_numpy(b).float())
    return linear
