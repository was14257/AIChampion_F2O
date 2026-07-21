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
        raise FileNotFoundError(f"RETFound 가중치 없음: {RETFOUND_W}")
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

    def __init__(self, freeze_encoder=False):
        super().__init__()
        self.encoder = _load_encoder()
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Dropout(0.3), nn.Linear(1024, 1))
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        feat = self.encoder.forward_features(x)[:, 0]
        return self.head(feat).squeeze(1)

    def param_groups(self, lr_enc, lr_head):
        return [
            {"params": self.encoder.parameters(), "lr": lr_enc},
            {"params": self.head.parameters(), "lr": lr_head},
        ]
