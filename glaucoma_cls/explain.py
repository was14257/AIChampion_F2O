"""Two explainability paths for the GlaucomaNet (RETFound embedding + 9
concepts) classifier.

Neither is involved in computing risk; they show "why this decision came
out" in two different ways (both are post-hoc methods needing no training):

  1) concept_saliency(): of the 9 concepts (cdr, ovality, thickness, etc.),
     how much each contributed to this case's risk (gradient*input). Even
     after passing through concept_proj (an MLP), gradient still flows back
     to the raw concept via the chain rule, so we set requires_grad on the
     standardized concept input to capture it. -> "numeric evidence"
  2) ViTGradCAM: a heatmap of where RETFound (ViT) looked in the fundus.
     patch token gradient*activation w.r.t. the risk logit. -> "spatial
     evidence"

Both paths attach to the single GlaucomaNet. Since forward takes the form
model(x, concepts), both Grad-CAM and saliency feed in x and concept
together to reproduce the actual risk logit.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glaucoma_cls.concepts import ALL_CONCEPTS


def mc_dropout_ci(model, x: torch.Tensor, concepts: torch.Tensor,
                  n: int = 30, lo: float = 2.5, hi: float = 97.5) -> dict:
    """Computes a confidence interval for the risk probability via MC-dropout.

    Running forward n times with dropout kept on even at inference (train
    mode) randomly kills a different slice of the embedding each time, so
    predictions vary slightly. We compute mean/CI/std from that
    distribution. Our model only has dropout on the RETFound embedding
    (section 16), so this CI reflects "uncertainty in the image feature
    (embedding) path" (concept is fixed, no dropout).

    Returns: {"mean", "lo", "hi", "std", "samples"(list)}.
    A narrow interval means high confidence, a wide one means uncertainty.
    """
    was_training = model.training
    model.eval()                       # keep LayerNorm etc. fixed in eval
    # Only switch dropout modules to train mode (the core of MC-dropout).
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()

    probs = []
    with torch.no_grad():
        for _ in range(n):
            p = torch.sigmoid(model(x, concepts)).item()
            probs.append(p)

    if was_training:
        model.train()
    else:
        model.eval()

    probs = np.array(probs)
    return {
        "mean": float(probs.mean()),
        "lo": float(np.percentile(probs, lo)),
        "hi": float(np.percentile(probs, hi)),
        "std": float(probs.std()),
        "samples": probs.tolist(),
    }


def concept_saliency(model, x: torch.Tensor, concepts: torch.Tensor) -> dict:
    """Each concept's contribution to risk in this case (gradient*input saliency).

    x: (1,3,224,224) fundus tensor, concepts: (1, 9) standardized concept vector.
    Returns: {concept name: contribution float} (positive=risk direction,
    negative=protective direction).
    """
    model.eval()
    c = concepts.clone().detach().requires_grad_(True)
    model.zero_grad()
    logit = model(x, c)          # risk logit (including the concept_proj pass)
    logit.sum().backward()
    contrib = (c.grad * c).detach().cpu().numpy()[0]   # (9,)
    return dict(zip(ALL_CONCEPTS, contrib.tolist()))


class ViTGradCAM:
    """Grad-CAM specific to GlaucomaNet. Hooks CAM onto the patch tokens of layer block_idx.

    The last block (-1) can have its gradient die to 0 because timm's
    forward_features entangles it with the path through the final norm to
    extract CLS (observed 2026-07-22), so the second-to-last (-2) is used
    as the default.
    """

    def __init__(self, model, block_idx: int = -2):
        self.model = model
        self.block = model.encoder.blocks[block_idx]
        self._activations = None
        self._gradients = None
        self._fh = self.block.register_forward_hook(self._save_activation)
        self._bh = self.block.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out          # (B, 1+N, D) - CLS + patch tokens

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, x: torch.Tensor, concepts: torch.Tensor) -> tuple[np.ndarray, float]:
        """x: (1,3,224,224), concepts: (1,9). Returns: (14,14) CAM(0~1), risk probability."""
        self.model.eval()
        self.model.zero_grad()
        logit = self.model(x, concepts)
        prob = torch.sigmoid(logit).item()
        logit.sum().backward()

        act = self._activations[0, 1:]      # (N, D) - patch tokens only (CLS excluded)
        grad = self._gradients[0, 1:]       # (N, D)
        weights = grad.mean(dim=0)          # (D,) - Grad-CAM channel weights (GAP)
        cam = F.relu((act * weights).sum(dim=-1))  # (N,)

        n_side = int(cam.numel() ** 0.5)
        cam = cam.reshape(n_side, n_side).detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, prob


def attention_rollout(model, x: torch.Tensor, concepts: torch.Tensor,
                      discard_ratio: float = 0.9, head_fusion: str = "mean",
                      last_n: int = 6, border_n: int = 3) -> tuple:
    """Attention Rollout (Abnar & Zuidema 2020) — shows where the ViT looks
    without using gradients. Adds a residual (identity) to each block's
    attention, normalizes, then accumulates across layers via matrix
    multiplication to get the CLS token's final attention to each patch.

    timm ViT uses fused SDPA so attention weights aren't exposed, so we hook
    each attn module's qkv output and recompute attention directly (no model
    changes).

    discard_ratio: discards this fraction of low attention at each layer to
    reduce noise.
    head_fusion: how to merge multi-head attention ("mean"/"max").
    last_n: use only the last n blocks for rollout (ViT-L has 24 blocks).
    Accumulating all 24 blocks mixes in earlier (still local) layers,
    spreading the signal and blurring the disc-area signal (2026-07-27,
    G1020 6-image comparison: all24 vs last6/12/max-fusion). last6 most
    consistently concentrates near disc/cup and is adopted as the default.
    border_n: after computing rollout, zeroes out the outer border_n rows of
    the 14x14 patch grid and renormalizes. Even with last6, strong residual
    artifacts remained at the corners of the circular mask; checking the raw
    14x14 (pre-masking) stage showed edge patches coming out 6-22x higher
    than center patches - a structural bias in the residual accumulation
    process itself that _fundus_mask (which only removes the black
    background outside the circle) can't filter (a well-known rollout
    property where weaker-attention patches get proportionally more identity
    residual weight). border_n=2 (2026-07-27 first validation, G1020 6
    images) was insufficient in some cases - in one REFUGE image, a strong
    signal remaining on the 3rd row (just inside the border) after
    suppression became the renormalization baseline and the corner artifact
    reappeared. Re-validated with border_n=3 (same G1020 6 images + that
    REFUGE image): the G1020 disc signal is preserved while the REFUGE
    image's corner artifact is greatly reduced (not fully eliminated - seems
    to be an extreme case). border_n=4 is nearly identical to 3, so no need
    to go more aggressive. 0 disables suppression (old behavior).
    Returns: (14,14) rollout map (0~1), risk probability.
    """
    model.eval()
    enc = model.encoder
    attn_mats = []
    handles = []

    def _hook(attn_module):
        def fn(module, inp, out):
            # Recompute attention from the qkv Linear's output
            xin = inp[0]                      # (B, N, C)
            B, N, C = xin.shape
            qkv = module.qkv(xin).reshape(B, N, 3, module.num_heads,
                                          module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            a = (q * module.scale) @ k.transpose(-2, -1)   # (B, heads, N, N)
            a = a.softmax(dim=-1)
            attn_mats.append(a.detach())
        return fn

    for blk in enc.blocks:
        handles.append(blk.attn.register_forward_hook(_hook(blk.attn)))

    with torch.no_grad():
        logit = model(x, concepts)
        prob = torch.sigmoid(logit).item()

    for h in handles:
        h.remove()

    # rollout: for each layer's attention, head fusion -> residual+normalize -> cumulative product
    mats = attn_mats if last_n is None else attn_mats[-last_n:]
    device = mats[0].device
    N = mats[0].shape[-1]
    result = torch.eye(N, device=device)
    for a in mats:
        a = a.max(dim=1).values if head_fusion == "max" else a.mean(dim=1)  # (B,N,N)
        a = a[0]
        # discard low attention (remove the bottom discard_ratio, excluding the CLS row)
        flat = a.view(-1)
        n_keep = int(flat.numel() * (1 - discard_ratio))
        if 0 < n_keep < flat.numel():
            thr_v = torch.kthvalue(flat, flat.numel() - n_keep).values
            a = torch.where(a >= thr_v, a, torch.zeros_like(a))
        a = a + torch.eye(N, device=device)    # residual
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    # attention the CLS (index 0) token gives to each patch
    cls_to_patch = result[0, 1:]               # (N_patch,)
    n_side = int(cls_to_patch.numel() ** 0.5)
    roll = cls_to_patch.reshape(n_side, n_side).cpu().numpy()
    if border_n > 0:
        roll[:border_n, :] = 0
        roll[-border_n:, :] = 0
        roll[:, :border_n] = 0
        roll[:, -border_n:] = 0
    if roll.max() > 0:
        roll = roll / roll.max()
    return roll, prob


def _fundus_mask(base_rgb: np.ndarray, thr: float = 0.06) -> np.ndarray:
    """A mask that is True only inside the fundus circle. Since the
    background (black margin) has low brightness, we estimate the circle via
    a brightness threshold (fits the actual captured circle better than a
    geometric circle). Only the background is removed regardless of the
    fundus size/position per image."""
    gray = base_rgb.astype(np.float32).mean(axis=2) / 255.0
    return gray > thr


def overlay_heatmap(img_pil, cam: np.ndarray, alpha: float = 0.45,
                    mask_background: bool = True):
    """Upsamples cam(14,14, 0~1) to the original size and produces an image
    with the heatmap overlaid.

    If mask_background=True, forces the CAM outside the fundus circle
    (black background) to 0, removing background artifacts the ViT produces
    at image borders/corners (so the explanation only comes from within the
    retina region). Renormalizes to 0~1 after zeroing the background.
    """
    import matplotlib.cm as mcm
    from PIL import Image

    W, H = img_pil.size
    base = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)

    cam_t = torch.from_numpy(cam)[None, None]
    cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
    cam_up = cam_up[0, 0].numpy()

    if mask_background:
        mask = _fundus_mask(base)
        cam_up = cam_up * mask
        if cam_up.max() > 0:
            cam_up = cam_up / cam_up.max()   # renormalize to 0~1 within the circle

    heat = (mcm.jet(cam_up)[:, :, :3] * 255).astype(np.uint8)
    blended = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    if mask_background:
        # leave the background as-is without overlaying the heatmap (also removes the blue background)
        blended = np.where(mask[..., None], blended, base).astype(np.uint8)
    return Image.fromarray(blended)
