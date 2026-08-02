from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


def _interpolate_pos_embed(encoder: nn.Module, state: dict) -> dict:
    if "pos_embed" not in state:
        return state
    ckpt_pe = state["pos_embed"]
    model_pe = encoder.pos_embed
    if ckpt_pe.shape == model_pe.shape:
        return state

    dim = ckpt_pe.shape[-1]
    num_patches = encoder.patch_embed.num_patches
    num_extra = model_pe.shape[1] - num_patches

    orig = int(round((ckpt_pe.shape[1] - num_extra) ** 0.5))
    new = int(round(num_patches ** 0.5))

    extra = ckpt_pe[:, :num_extra]
    grid = ckpt_pe[:, num_extra:].reshape(1, orig, orig, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new, new), mode="bicubic",
                         align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new * new, dim)
    state["pos_embed"] = torch.cat([extra, grid], dim=1)
    print(f"[RETFound] pos_embed interpolated: {orig}x{orig} -> {new}x{new}")
    return state


def load_retfound_encoder(weights_path: Path | None = None,
                          verbose: bool = True) -> nn.Module:
    mcfg = CFG.model
    encoder = timm.create_model(
        mcfg.backbone,
        pretrained=False,
        num_classes=0,
        img_size=CFG.data.img_size,
    )

    if weights_path is None:
        weights_path = CFG.paths.retfound_weights

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"RETFound weights not found: {weights_path}\n"
            "See RETFound_DOWNLOAD.md to obtain them and place them at that path."
        )

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("model", "teacher", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break

    cleaned = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "encoder.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        if nk.startswith(("head", "decoder", "mask_token", "fc_norm")):
            continue
        cleaned[nk] = v

    cleaned = _interpolate_pos_embed(encoder, cleaned)

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    if verbose:
        print(f"[RETFound] loaded from {weights_path}")
        print(f"[RETFound] matched {len(cleaned) - len(unexpected)} tensors, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if unexpected:
            print(f"[RETFound] unexpected (first 5): {unexpected[:5]}")
    return encoder


class _FuseBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.reduce(x))


class RetFoundSegmenter(nn.Module):
    def __init__(self,
                 encoder: nn.Module | None = None,
                 load_weights: bool = True):
        super().__init__()
        self.mcfg = CFG.model
        self.dcfg = CFG.data

        if encoder is None:
            encoder = (load_retfound_encoder() if load_weights
                       else timm.create_model(
                           self.mcfg.backbone, pretrained=False,
                           num_classes=0, img_size=self.dcfg.img_size))
        self.encoder = encoder

        if self.mcfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.n_blocks = len(self.encoder.blocks)
        self.feature_idx = [l - 1 for l in self.mcfg.feature_layers]

        dd = self.mcfg.decoder_dim
        self.fuses = nn.ModuleList(
            [_FuseBlock(self.mcfg.embed_dim, dd)
             for _ in self.feature_idx]
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dd * len(self.feature_idx), dd, 3, padding=1, bias=False),
            nn.BatchNorm2d(dd),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            self._up(dd, dd),
            self._up(dd, dd // 2),
            self._up(dd // 2, dd // 4),
            self._up(dd // 4, dd // 4),
        )
        self.head = nn.Conv2d(dd // 4, self.dcfg.num_classes, kernel_size=1)

    @staticmethod
    def _up(cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = self.encoder.get_intermediate_layers(
            x,
            n=self.feature_idx,
            reshape=True,
            norm=True,
        )
        fused = [fuse(f) for fuse, f in zip(self.fuses, feats)]
        x = torch.cat(fused, dim=1)
        x = self.fuse_conv(x)
        x = self.up(x)
        logits = self.head(x)
        if logits.shape[-2:] != (self.dcfg.img_size, self.dcfg.img_size):
            logits = F.interpolate(
                logits, size=(self.dcfg.img_size, self.dcfg.img_size),
                mode="bilinear", align_corners=False)
        return logits

    def param_groups(self):
        enc, dec = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if n.startswith("encoder.") else dec).append(p)
        return enc, dec


def build_model(load_weights: bool = True) -> RetFoundSegmenter:
    return RetFoundSegmenter(load_weights=load_weights)
