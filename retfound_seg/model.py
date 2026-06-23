"""
RETFound(ViT-Large/16, MAE 사전학습) 인코더 + 세그멘테이션 디코더.

RETFound 자체는 분할 모델이 아니라 ViT 인코더(특징추출기)이므로,
여기에 다중 스케일 디코더(SETR-MLA 계열)를 붙여 disc/cup 3-class 분할을 수행한다.

인코더 가중치 로딩은 RETFound 공식 체크포인트(.pth, 'model' 키)와
HuggingFace 변환본을 모두 처리하도록 작성했다.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from local_config import CFG


# --------------------------------------------------------------------------- #
# 위치 임베딩 보간 (사전학습 224 -> 목표 해상도)                                  #
# --------------------------------------------------------------------------- #
def _interpolate_pos_embed(encoder: nn.Module, state: dict) -> dict:
    """체크포인트의 pos_embed(224 격자)를 현재 입력 해상도 격자로 보간한다.

    RETFound 는 14x14(=196 토큰)로 학습됐는데, 입력을 512 로 키우면
    32x32(=1024 토큰)가 필요하므로 bicubic 보간으로 맞춰준다.
    """
    if "pos_embed" not in state:
        return state
    ckpt_pe = state["pos_embed"]                       # [1, N_ckpt, C]
    model_pe = encoder.pos_embed                       # [1, N_model, C]
    if ckpt_pe.shape == model_pe.shape:
        return state                                   # 224 그대로면 보간 불필요

    dim = ckpt_pe.shape[-1]
    num_patches = encoder.patch_embed.num_patches
    num_extra = model_pe.shape[1] - num_patches        # cls 토큰 등(보통 1)

    orig = int(round((ckpt_pe.shape[1] - num_extra) ** 0.5))
    new = int(round(num_patches ** 0.5))

    extra = ckpt_pe[:, :num_extra]
    grid = ckpt_pe[:, num_extra:].reshape(1, orig, orig, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new, new), mode="bicubic",
                         align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new * new, dim)
    state["pos_embed"] = torch.cat([extra, grid], dim=1)
    print(f"[RETFound] pos_embed 보간: {orig}x{orig} -> {new}x{new}")
    return state


# --------------------------------------------------------------------------- #
# RETFound 인코더 가중치 로더                                                    #
# --------------------------------------------------------------------------- #
def load_retfound_encoder(weights_path: Path | None = None,
                          verbose: bool = True) -> nn.Module:
    """timm ViT-L/16 을 만들고 RETFound 사전학습 가중치를 적재한다.

    반환된 모델은 분류 헤드 없이 (num_classes=0) 토큰 특징만 내보낸다.
    """
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
            f"RETFound 가중치를 찾을 수 없음: {weights_path}\n"
            "RETFound_DOWNLOAD.md 를 보고 받아서 해당 경로에 두세요."
        )

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    # 체크포인트 형식별로 state_dict 추출
    state = ckpt
    for key in ("model", "teacher", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break

    # 접두사 정리 및 분류 헤드 제거
    cleaned = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "encoder.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        # MAE/분류 헤드, 디코더 관련 키는 분할에 불필요 -> 버림
        if nk.startswith(("head", "decoder", "mask_token", "fc_norm")):
            continue
        cleaned[nk] = v

    # 입력 해상도가 224 가 아니면 pos_embed 를 보간해 맞춘다.
    cleaned = _interpolate_pos_embed(encoder, cleaned)

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    if verbose:
        print(f"[RETFound] loaded from {weights_path}")
        print(f"[RETFound] matched {len(cleaned) - len(unexpected)} tensors, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if unexpected:
            print(f"[RETFound] unexpected(앞5): {unexpected[:5]}")
    return encoder


# --------------------------------------------------------------------------- #
# 디코더: 선택한 트랜스포머 층의 특징을 융합해 점진적 업샘플                       #
# --------------------------------------------------------------------------- #
class _FuseBlock(nn.Module):
    """1x1 conv 로 채널 축소 후 conv 정제."""

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
    """RETFound ViT 인코더 + 다중층 융합 디코더."""

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

        # get_intermediate_layers 에 넘길 "마지막 n개 층" 개수.
        # feature_layers 가 (6,12,18,24) 라면 24개 블록 중 해당 인덱스를 고른다.
        self.n_blocks = len(self.encoder.blocks)
        self.feature_idx = [l - 1 for l in self.mcfg.feature_layers]  # 0-based

        dd = self.mcfg.decoder_dim
        self.fuses = nn.ModuleList(
            [_FuseBlock(self.mcfg.embed_dim, dd)
             for _ in self.feature_idx]
        )
        # 융합된 특징을 합친 뒤 점진적 업샘플 (14 -> 224, x16)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dd * len(self.feature_idx), dd, 3, padding=1, bias=False),
            nn.BatchNorm2d(dd),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            self._up(dd, dd),          # x2
            self._up(dd, dd // 2),     # x4
            self._up(dd // 2, dd // 4),  # x8
            self._up(dd // 4, dd // 4),  # x16
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
        # timm VisionTransformer: 선택한 블록의 출력을 [B, C, H, W] 로 받음
        feats = self.encoder.get_intermediate_layers(
            x,
            n=self.feature_idx,         # 인덱스 리스트 지원 (timm 최신)
            reshape=True,
            norm=True,
        )
        fused = [fuse(f) for fuse, f in zip(self.fuses, feats)]
        x = torch.cat(fused, dim=1)
        x = self.fuse_conv(x)
        x = self.up(x)
        logits = self.head(x)
        # 입력 해상도에 맞춰 보정
        if logits.shape[-2:] != (self.dcfg.img_size, self.dcfg.img_size):
            logits = F.interpolate(
                logits, size=(self.dcfg.img_size, self.dcfg.img_size),
                mode="bilinear", align_corners=False)
        return logits

    # 인코더/디코더 파라미터를 분리 (서로 다른 LR 적용용)
    def param_groups(self):
        enc, dec = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if n.startswith("encoder.") else dec).append(p)
        return enc, dec


def build_model(load_weights: bool = True) -> RetFoundSegmenter:
    return RetFoundSegmenter(load_weights=load_weights)
