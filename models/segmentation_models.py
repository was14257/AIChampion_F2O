"""
두 모델 정의:
  1. RETFoundSegmenter  : frozen(or fine-tuned) RETFound ViT-L 인코더 +
                          Segmenter 스타일 mask-transformer decoder
  2. UNetBaseline       : 처음부터 학습하는 표준 U-Net (fair comparison용)

클래스 수 K=3 (background=0, OD-ring(cup 제외한 disc)=1, OC=2) 로 고정.
평가 시 "전체 OD" = (pred==1) | (pred==2) 로 재구성해서 Dice 계산.
"""
import torch
import torch.nn as nn

from .retfound_backbone import RETFoundEncoder

NUM_CLASSES = 3  # background, OD-ring, OC


# ============================================================
# 1) RETFound + Segmenter-style mask transformer decoder
# ============================================================
class MaskTransformerDecoder(nn.Module):
    """Segmenter(Strudel et al., 2021) 스타일 경량 decoder.

    patch token들과 학습 가능한 class mask token들을 함께
    shallow transformer encoder에 통과시킨 뒤, patch token과
    mask token의 내적으로 클래스별 per-patch score를 얻어.
    """

    def __init__(self, embed_dim: int, num_classes: int = NUM_CLASSES, depth: int = 2, num_heads: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.class_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, grid_size: int):
        # tokens: (B, N, C)
        B, N, C = tokens.shape
        cls_tok = self.cls_tokens.expand(B, -1, -1)  # (B, K, C)
        x = torch.cat([tokens, cls_tok], dim=1)  # (B, N+K, C)
        x = self.transformer(x)

        patches = self.patch_norm(x[:, :N])  # (B, N, C)
        classes = self.class_norm(x[:, N:])  # (B, K, C)
        patches = self.proj(patches)

        # per-patch, per-class score = 내적
        masks = torch.einsum("bnc,bkc->bnk", patches, classes)  # (B, N, K)
        masks = masks.transpose(1, 2).reshape(B, self.num_classes, grid_size, grid_size)
        return masks  # (B, K, H', W') -- 아직 원본 해상도 아님


class RETFoundSegmenter(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        checkpoint_path: str | None = None,
        freeze_encoder: bool = True,
        num_classes: int = NUM_CLASSES,
        decoder_depth: int = 2,
    ):
        super().__init__()
        self.encoder = RETFoundEncoder(
            img_size=img_size, checkpoint_path=checkpoint_path, freeze=freeze_encoder
        )
        self.decoder = MaskTransformerDecoder(
            embed_dim=self.encoder.embed_dim, num_classes=num_classes, depth=decoder_depth
        )
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.encoder(x)
        masks = self.decoder(tokens, self.encoder.grid_size)  # (B, K, 14, 14)
        masks = nn.functional.interpolate(
            masks, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        return masks  # logits, (B, K, H, W)


# ============================================================
# 2) 생 U-Net (from scratch, fair baseline)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBaseline(nn.Module):
    """표준 U-Net. base_ch=64 기준 파라미터 수는 RETFound(ViT-L, 300M+)보다
    훨씬 작음 -> 비교표에 파라미터 수도 같이 적어주는 걸 추천."""

    def __init__(self, in_ch: int = 3, num_classes: int = NUM_CLASSES, base_ch: int = 64):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.enc1 = DoubleConv(in_ch, chs[0])
        self.enc2 = DoubleConv(chs[0], chs[1])
        self.enc3 = DoubleConv(chs[1], chs[2])
        self.enc4 = DoubleConv(chs[2], chs[3])
        self.bottleneck = DoubleConv(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = DoubleConv(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[1], chs[0])

        self.out_conv = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)  # logits, (B, K, H, W)


def build_model(name: str, img_size: int = 224, retfound_ckpt: str | None = None, freeze_encoder: bool = True):
    name = name.lower()
    if name == "retfound":
        return RETFoundSegmenter(img_size=img_size, checkpoint_path=retfound_ckpt, freeze_encoder=freeze_encoder)
    elif name == "unet":
        return UNetBaseline()
    else:
        raise ValueError(f"Unknown model name: {name}")
