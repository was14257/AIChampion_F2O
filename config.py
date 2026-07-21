import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

import local_config as _lc


@dataclass(frozen=True)
class PathConfig:

    root: Path = Path(__file__).resolve().parent

    data_root: Path = _lc.DATA_ROOT
    output_root: Path = _lc.OUTPUT_ROOT

    use_cropped: bool = _lc.USE_CROPPED

    g1020: Path = _lc.G1020_ROOT
    grape: Path = _lc.GRAPE_ROOT
    origa: Path = _lc.ORIGA_ROOT
    refuge: Path = _lc.REFUGE_ROOT
    gamma: Path = _lc.GAMMA_ROOT
    gamma_grading: Path = _lc.GAMMA_GRADING_ROOT

    oct_labels: Path = _lc.OCT_LABELS_ROOT
    oct_pseudo: Path = _lc.OCT_PSEUDO_ROOT
    oct_features: Path = _lc.OCT_FEATURES_ROOT
    diffusion_out: Path = _lc.DIFFUSION_OUT_ROOT

    refuge_train_img: Path = field(init=False)
    refuge_train_mask: Path = field(init=False)
    refuge_val_img: Path = field(init=False)
    refuge_val_mask: Path = field(init=False)
    refuge_test_img: Path = field(init=False)
    refuge_test_mask: Path = field(init=False)

    models_dir: Path = _lc.Models_DIR
    retfound_weights: Path = _lc.RETFound_WEIGHTS

    output_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)
    pred_dir: Path = field(init=False)

    def __post_init__(self):
        s = object.__setattr__

        img_dir = "Images_Cropped" if self.use_cropped else "Images"
        mask_dir = "Masks_Cropped" if self.use_cropped else "Masks"
        s(self, "refuge_train_img", self.refuge / "train" / img_dir)
        s(self, "refuge_train_mask", self.refuge / "train" / mask_dir)
        s(self, "refuge_val_img", self.refuge / "val" / img_dir)
        s(self, "refuge_val_mask", self.refuge / "val" / mask_dir)
        s(self, "refuge_test_img", self.refuge / "test" / img_dir)
        s(self, "refuge_test_mask", self.refuge / "test" / mask_dir)

        s(self, "output_dir", self.output_root)
        s(self, "ckpt_dir", self.output_root / "checkpoints")
        s(self, "pred_dir", self.output_root / "predictions")

    def ensure_dirs(self) -> None:
        for p in (self.output_dir, self.models_dir, self.ckpt_dir, self.pred_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:

    num_classes: int = 3
    label_background: int = 0
    label_disc_rim: int = 1
    label_cup: int = 2

    img_size: int = _lc.IMG_SIZE

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    num_workers: int = _lc.num_workers
    pin_memory: bool = _lc.PIN_MEMORY


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "vit_large_patch16_224"
    embed_dim: int = 1024
    patch_size: int = 16

    feature_layers: tuple[int, int, int, int] = _lc.FEATURE_LAYERS
    decoder_dim: int = 256

    freeze_encoder: bool = _lc.FREEZE_ENCODER

    @property
    def encoder_weights(self) -> Path:
        return PathConfig().retfound_weights


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = _lc.EPOCHS
    batch_size: int = _lc.BATCH_SIZE
    lr_encoder: float = _lc.LR_ENCODER
    lr_decoder: float = _lc.lR
    weight_decay: float = _lc.WEIGHT_DECAY
    warmup_epochs: int = _lc.WARMUP_EPOCHS
    grad_accum: int = _lc.GRAD_ACCUM

    ce_weight: float = 1.0
    dice_weight: float = 1.0

    use_amp: bool = _lc.USE_AMP
    grad_clip: float = _lc.GRAD_CLIP
    seed: int = _lc.SEED

    monitor: str = "mean_dice"
    save_best_only: bool = True


@dataclass(frozen=True)
class InferConfig:
    checkpoint_name: str = _lc.CHECKPOINT_NAME

    cdr_kind: str = _lc.CDR_KIND

    keep_largest_cc: bool = _lc.KEEP_LARGEST_CC

    @property
    def checkpoint_path(self) -> Path:
        return PathConfig().ckpt_dir / self.checkpoint_name


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class OctTier1Config:
    horig: int = _lc.OCT_BSCAN_HORIG
    worig: int = _lc.OCT_BSCAN_WORIG

    seg_h: int = _lc.SEG_H
    seg_w: int = _lc.SEG_W

    diff_h: int = _lc.DIFF_H
    diff_w: int = _lc.DIFF_W
    diff_t: int = _lc.DIFF_T
    flatten_rpe: bool = _lc.FLATTEN_RPE

    diff_full_h: int = _lc.DIFF_FULL_H
    diff_full_w: int = _lc.DIFF_FULL_W
    diff_full_ch: int = _lc.DIFF_FULL_CH

    seg_run_mode: str = _lc.SEG_RUN_MODE
    seg_epochs: int = _lc.SEG_EPOCHS
    seg_predict_volumes: tuple | None = _lc.SEG_PREDICT_VOLUMES
    seg_slices_per_vol: int = _lc.SEG_SLICES_PER_VOL

    diff_run_mode: str = _lc.DIFF_RUN_MODE
    diff_epochs: int = _lc.DIFF_EPOCHS
    diff_bs: int = _lc.DIFF_BS
    diff_sample_n: int = _lc.DIFF_SAMPLE_N

    diff_full_root: Path = _lc.DIFF_FULL_ROOT
    diff_full_pseudo: Path = _lc.DIFF_FULL_PSEUDO
    diff_full_out: Path = _lc.DIFF_FULL_OUT
    diff_full_epochs: int = _lc.DIFF_FULL_EPOCHS
    diff_full_bs: int = _lc.DIFF_FULL_BS
    diff_full_lr: float = _lc.DIFF_FULL_LR
    diff_full_save_every: int = _lc.DIFF_FULL_SAVE_EVERY

    label_only_case: str | None = _lc.LABEL_ONLY_CASE
    pick_start_mid: int = _lc.PICK_START_MID
    review_volume: str | None = _lc.REVIEW_VOLUME
    review_include_all: bool = _lc.REVIEW_INCLUDE_ALL
    review_redo: bool = _lc.REVIEW_REDO
    laterality_review_all: bool = _lc.LATERALITY_REVIEW_ALL


@dataclass(frozen=True)
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    oct_tier1: OctTier1Config = field(default_factory=OctTier1Config)


CFG = Config()


if __name__ == "__main__":
    CFG.paths.ensure_dirs()
    print("device           :", CFG.runtime.device)
    print("root             :", CFG.paths.root)
    print("data_root        :", CFG.paths.data_root)
    print("output_root      :", CFG.paths.output_root)
    print("REFUGE train img :", CFG.paths.refuge_train_img,
          "(존재:", CFG.paths.refuge_train_img.exists(), ")")
    print("RETFound weights :", CFG.paths.retfound_weights,
          "(존재:", CFG.paths.retfound_weights.exists(), ")")
    print("checkpoint dir   :", CFG.paths.ckpt_dir)
    print()
    print("datasets:", json.dumps({
        "G1020": str(CFG.paths.g1020),
        "GRAPE": str(CFG.paths.grape),
        "ORIGA": str(CFG.paths.origa),
        "REFUGE": str(CFG.paths.refuge),
        "GAMMA": str(CFG.paths.gamma),
    }, indent=2, ensure_ascii=False))
    print("train:", json.dumps({
        "epochs": CFG.train.epochs,
        "batch_size": CFG.train.batch_size,
        "grad_accum": CFG.train.grad_accum,
        "lr_encoder": CFG.train.lr_encoder,
        "lr_decoder": CFG.train.lr_decoder,
        "num_workers": CFG.data.num_workers,
        "seed": CFG.train.seed,
    }, indent=2, ensure_ascii=False))
