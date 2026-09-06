import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from bscan_gen.utils import disc_crop, embed_fundus
from glaucoma_cls.concepts import (seg_concepts, OCT_CONCEPTS, ALL_CONCEPTS,  # noqa: F401
                                   RNFL_CONCEPTS, ALL_CONCEPTS_V2)
from retfound_seg.cdr import postprocess_label
from retfound_seg.model import RetFoundSegmenter


class EyeonCBM(nn.Module):
    """Inference wrapper bundling seg_model / oct_encoder+oct_linear to extract the 9 concepts.

    seg_model: retfound_seg.model.RetFoundSegmenter (loads existing weights, 512 input)
    oct_encoder: 224-input RETFound built via bscan_gen.utils.load_retfound_encoder()
    oct_linear: nn.Linear built via glaucoma_cls.concepts.fit_oct_linear() (whole+disc input)
    rnfl_model: optional dict {"scaler","pls","targets"} from grape_train_rnfl.py
        (fundus whole+disc embedding -> GRAPE RNFL Mean/S/N/I/T, sklearn PLS).
        When given, predict_from_path() returns 11 concepts (SEG 6 + RNFL 5)
        instead of the original 9 (SEG 6 + OCT_CONCEPTS 3).
    """

    def __init__(self, seg_model: RetFoundSegmenter, oct_encoder: nn.Module,
                 oct_linear: nn.Linear, disc_x_fallback_ratio: float = 0.5,
                 rnfl_model: dict | None = None):
        """Stores the seg/oct sub-models and precomputes the RNFL output index order."""
        super().__init__()
        self.seg_model = seg_model
        self.oct_encoder = oct_encoder
        self.oct_linear = oct_linear
        self.disc_x_fallback_ratio = disc_x_fallback_ratio
        self.rnfl_model = rnfl_model
        if rnfl_model is not None:
            targets = rnfl_model["targets"]  # e.g. [mean_th, S, N, I, T]
            self._rnfl_order_idx = [targets.index("mean_th"), targets.index("I"),
                                    targets.index("S"), targets.index("N"), targets.index("T")]

    def _disc_x_from_mask(self, label_map: np.ndarray, img_w: int) -> int:
        """Finds the disc's horizontal center in original-image pixel coords (fallback if no disc found)."""
        disc = label_map >= 1
        if disc.sum() == 0:
            return int(img_w * self.disc_x_fallback_ratio)
        cols = np.where(disc.any(axis=0))[0]
        seg_w = label_map.shape[1]
        return int((cols.min() + cols.max()) / 2 / seg_w * img_w)

    @torch.no_grad()
    def predict_from_path(self, img_path: str, device: str = "cpu") -> dict:
        """One fundus image path -> {concept name: value} (9 total)."""
        img = Image.open(img_path).convert("RGB")

        # --- pass 1: whole -> segmentation -> 6 geometric concepts ---
        seg_input = _preprocess_from_image(img, device)
        logits = self.seg_model(seg_input)
        label_map = postprocess_label(logits.argmax(dim=1)[0].cpu().numpy())
        seg_out = seg_concepts(label_map)

        # --- pass 2/3: disc crop -> whole+disc embedding (224 encoder) -> OCT concepts ---
        dx = self._disc_x_from_mask(label_map, img.width)
        cropped = disc_crop(img, dx)
        whole_emb = embed_fundus(self.oct_encoder, img)
        disc_emb = embed_fundus(self.oct_encoder, cropped)
        dual = np.concatenate([whole_emb, disc_emb])[None, :]

        if self.rnfl_model is not None:
            pred = self.rnfl_model["pls"].predict(
                self.rnfl_model["scaler"].transform(dual))[0]
            pred = pred[self._rnfl_order_idx]
            oct_out = dict(zip(RNFL_CONCEPTS, pred.tolist()))
        else:
            oct_pred = self.oct_linear(torch.from_numpy(dual).float().to(device))[0]
            oct_out = dict(zip(OCT_CONCEPTS, oct_pred.tolist()))

        return {**seg_out, **oct_out}


def _preprocess_from_image(img: Image.Image, device):
    """Resizes/normalizes a PIL image into the seg model's input tensor."""
    from config import CFG
    size = CFG.data.img_size
    arr = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CFG.data.mean).view(3, 1, 1)
    std = torch.tensor(CFG.data.std).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)
