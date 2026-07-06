"""
RETFound (ViT-Large/16) 인코더 래퍼.

RETFound는 fundus/OCT 이미지로 self-supervised(MAE) 사전학습된
ViT-Large 백본이야 (hidden=1024, depth=24, heads=16, patch=16, img=224).
공식 체크포인트: https://github.com/rmaphoh/RETFound

이 클래스는:
  1. timm으로 표준 ViT-Large/16 아키텍처를 만들고
  2. RETFound 체크포인트(state_dict)를 최대한 매칭해서 로드하고
  3. segmentation decoder가 쓸 수 있도록 CLS 토큰을 제외한
     patch token sequence (B, N, C)와 spatial map (B, C, H', W')을 반환해.
"""
import torch
import torch.nn as nn
import timm


class RETFoundEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        checkpoint_path: str | None = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 224/16 = 14

        # num_classes=0, global_pool='' -> forward_features가 (B, N+1, C) 토큰 시퀀스를 반환
        self.vit = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=img_size,
            num_classes=0,
            global_pool="",
        )
        self.embed_dim = self.vit.embed_dim  # 1024

        if checkpoint_path is not None:
            self.load_retfound_weights(checkpoint_path)

        self._frozen = freeze
        if freeze:
            self.freeze_encoder()

    # ------------------------------------------------------------------
    def load_retfound_weights(self, checkpoint_path: str):
        """RETFound 공식 체크포인트를 최대한 매칭해서 로드.

        체크포인트는 보통 {'model': state_dict, ...} 형태로 저장돼 있어
        (MAE 계열 코드베이스 관례). state_dict 키가 timm VisionTransformer랑
        완전히 동일하지 않을 수 있어서 strict=False로 로드하고
        missing/unexpected 키를 출력해줘 -> 실제 환경에서 확인하고 필요하면
        키 이름 매핑을 추가해줘 (예: 'blocks.0.attn.qkv.weight' 등은 보통 동일함).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        # 분류 head, decoder(MAE reconstruction용) 관련 키는 어차피 필요 없으니 제거
        drop_prefixes = ("head.", "decoder_", "mask_token")
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith(drop_prefixes)
        }

        # 위치 임베딩 크기가 안 맞으면(입력 해상도가 사전학습과 다르면) 보간
        if "pos_embed" in state_dict:
            state_dict["pos_embed"] = self._interpolate_pos_embed(
                state_dict["pos_embed"]
            )

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        print(f"[RETFound] loaded checkpoint: {checkpoint_path}")
        print(f"[RETFound] missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        print(f"[RETFound] unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    def _interpolate_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        target_num_patches = self.grid_size**2
        num_extra_tokens = self.vit.pos_embed.shape[1] - self.grid_size**2  # 보통 CLS 1개
        src_num_patches = pos_embed.shape[1] - num_extra_tokens
        if src_num_patches == target_num_patches:
            return pos_embed

        src_size = int(src_num_patches**0.5)
        dst_size = self.grid_size
        extra_tokens = pos_embed[:, :num_extra_tokens]
        patch_pos = pos_embed[:, num_extra_tokens:]
        dim = patch_pos.shape[-1]

        patch_pos = patch_pos.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
        patch_pos = torch.nn.functional.interpolate(
            patch_pos, size=(dst_size, dst_size), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, dst_size * dst_size, dim)
        return torch.cat([extra_tokens, patch_pos], dim=1)

    # ------------------------------------------------------------------
    def freeze_encoder(self):
        self._frozen = True
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        self._frozen = False
        for p in self.vit.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Returns:
            tokens: (B, N, C)  -- CLS 제외한 patch token sequence (decoder 입력용)
            spatial: (B, C, H', W') -- 시각화/conv decoder용 spatial feature map
        """
        if self._frozen:
            # frozen encoder는 gradient가 필요 없으니 no_grad로 감싸서
            # activation 메모리를 아낌 (맥북처럼 메모리 빠듯한 로컬 환경 배려)
            with torch.no_grad():
                feats = self.vit.forward_features(x)  # (B, N+1, C), 인덱스 0 = CLS
        else:
            feats = self.vit.forward_features(x)
        tokens = feats[:, 1:, :]  # CLS 제거
        B, N, C = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)
        return tokens, spatial
