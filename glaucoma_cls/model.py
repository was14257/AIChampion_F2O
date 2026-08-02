import timm
import torch
import torch.nn as nn

from glaucoma_cls.data import DATA

RETFOUND_W = DATA / "models/RETFound_cfp_weights.pth"
IMG_SIZE = 224


def _load_encoder():
    enc = timm.create_model("vit_large_patch16_224", pretrained=False,
                            num_classes=0, img_size=IMG_SIZE)
    if not RETFOUND_W.exists():
        raise FileNotFoundError(f"RETFound weights not found: {RETFOUND_W}")
    ckpt = torch.load(RETFOUND_W, map_location="cpu", weights_only=False)
    state = ckpt
    for k in ("model", "teacher", "state_dict"):
        if isinstance(ckpt, dict) and k in ckpt:
            state = ckpt[k]
            break
    clean = {}
    for k, v in state.items():
        nk = k
        for pre in ("module.", "encoder.", "backbone."):
            if nk.startswith(pre):
                nk = nk[len(pre):]
        if nk.startswith(("head", "decoder", "mask_token", "fc_norm")):
            continue
        clean[nk] = v
    miss, unexp = enc.load_state_dict(clean, strict=False)
    print(f"[RETFound-cls] matched={len(clean)-len(unexp)} missing={len(miss)} unexpected={len(unexp)}")
    return enc


class GlaucomaNet(nn.Module):

    def __init__(self, freeze_encoder=False, unfreeze_last_n=0, n_concepts=0,
                concept_proj_dim=0):
        """freeze_encoder=True: freezes the entire encoder.
        unfreeze_last_n>0: with freeze_encoder=True, re-unfreezes only the
        last N blocks (+final norm) - a compromise between full fine-tuning
        (slow, overfitting risk with only 172~2070 samples) and fully frozen
        (head only, insufficient expressiveness).
        n_concepts>0: concatenates concept values (CDR/ilm_rough etc.) with
        the RETFound embedding (1024) and feeds both into the head (not a
        pure CBM using concepts alone - keeps the embedding info and adds
        concepts as an auxiliary signal).
        concept_proj_dim>0: concatenating concept (n_concepts-dim, usually 9)
        raw is overwhelmingly small relative to the embedding (1024-dim),
        raising concern about asymmetric dropout/weight_decay, so a small
        dedicated concept MLP expands it to concept_proj_dim before
        concatenation (an experimental option to increase concept's relative weight)."""
        super().__init__()
        self.encoder = _load_encoder()
        self.n_concepts = n_concepts
        self.concept_proj_dim = concept_proj_dim
        # Mixing embedding (1024) and concept (n_concepts) into a single
        # LayerNorm tangles the two signals' scales (concept is already
        # standardized), destabilizing training (measured: combined with
        # pos_weight correction, collapses to sens=1.0/spec=0, macro_f1
        # plummets). LayerNorm the embedding only, and concat concept as-is
        # to keep them separate.
        self.feat_norm = nn.LayerNorm(1024)
        # Dropout is applied only to the RETFound embedding (1024). Concept
        # is already a standardized low-dim signal (9~32) - dropping it
        # wholesale would randomly wipe out key concepts like CDR and
        # destabilize training (also inconsistent with improvement 6's intent
        # of boosting concept's weight via concept_proj). MC-dropout at
        # inference also only reflects embedding-side uncertainty.
        self.feat_dropout = nn.Dropout(0.3)
        if n_concepts > 0 and concept_proj_dim > 0:
            self.concept_proj = nn.Sequential(
                nn.Linear(n_concepts, concept_proj_dim), nn.ReLU())
            concept_out_dim = concept_proj_dim
        else:
            self.concept_proj = None
            concept_out_dim = n_concepts
        self.head = nn.Linear(1024 + concept_out_dim, 1)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            if unfreeze_last_n > 0:
                for blk in self.encoder.blocks[-unfreeze_last_n:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                for p in self.encoder.norm.parameters():
                    p.requires_grad_(True)

    def forward(self, x, concepts=None):
        feat = self.encoder.forward_features(x)[:, 0]
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)   # dropout applied only to the RETFound embedding
        if self.n_concepts > 0:
            c = self.concept_proj(concepts) if self.concept_proj is not None else concepts
            feat = torch.cat([feat, c], dim=1)
        return self.head(feat).squeeze(1)

    def param_groups(self, lr_enc, lr_head):
        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self.feat_norm.parameters()) + list(self.head.parameters())
        if self.concept_proj is not None:
            head_params += list(self.concept_proj.parameters())
        return [
            {"params": enc_params, "lr": lr_enc},
            {"params": head_params, "lr": lr_head},
        ]
