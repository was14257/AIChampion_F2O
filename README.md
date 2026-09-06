# EYEON — Fundus-Derived OCT Structure & Glaucoma Screening

Given a single color fundus photograph (CFP), this project (1) generates a synthetic macular
OCT B-scan, (2) regresses structural OCT features, (3) classifies glaucoma risk directly from
the fundus image, and (4) explains that risk with concept-based saliency. It combines a
generative track (conditional DDPM) and a classification track (RETFound + concept
bottleneck), served together from a single Streamlit demo.

---

## Pipelines

1. **`retfound_seg`** — segments optic disc/cup from fundus and computes the C/D ratio.
2. **`glaucoma_cls`** — RETFound embedding + 11 clinical concepts (segmentation geometry +
   RNFL thickness) concatenated into a single linear head, trained for glaucoma classification.
   OOF AUROC 0.960 on a pooled GAMMA+REFUGE+GRAPE dataset (n=763). The head is kept linear
   (not MLP) so `concept_saliency` (gradient × input) is a rigorous per-concept contribution,
   which is the project's main explainability tool — attention/Grad-CAM heatmaps are shown for
   reference only, since they were found unreliable as spatial evidence.
3. **`bscan_gen`** — Stage A predicts a macular thickness profile from the fundus embedding
   (RETFound frozen + PLS); Stage B renders that profile into a realistic OCT B-scan with a
   from-scratch Palette-style conditional DDPM (channel-concat conditioning, not ControlNet).
4. **`feature_extraction`** — preprocessing scripts that build the cached concept/embedding
   files the two model tracks above load (`cbm_concepts_v2.npz`, `grape_rnfl_pls.pkl`, etc.).

All four are served together by [`app_streamlit.py`](app_streamlit.py) — upload a fundus image
and get OOD gating, disc/cup overlay, glaucoma risk with confidence interval, a synthetic OCT
B-scan, an 11-concept table, and a PDF report in one screen.

> Known limitation (see `../CLAUDE.md` for the full investigation): the OCT generation pipeline
> reproduces average macular shape well but only ~20-24% of individual variation. Extensive
> ablation traced this to a data modality limit — a fundus photo carries almost no signal about
> the eye's absolute RPE tilt during OCT acquisition (fundus→slope PLS OOF r²=0.014) — rather
> than a bug in the renderer or the nearest-neighbor shape search.

---

## Configuration

**All run parameters live in [`local_config.py`](local_config.py). Scripts take no CLI flags** —
edit the relevant constants there, then run the module.

- [`local_config.py`](local_config.py) — actual values: paths, hyperparameters, run modes.
- [`config.py`](config.py) — wraps `local_config` into typed `@dataclass` groups exposed as `CFG`
  (`from config import CFG`).

Data/output roots auto-detect local (Windows) vs. server (`/home/tta`) environments.

---

## Layout

```
Codes/
├─ local_config.py          # all settings (paths / hyperparameters / run modes)
├─ config.py                # local_config -> typed CFG dataclasses
├─ requirements.txt
├─ gamma_preview.py         # quick fundus <-> central B-scan pairing preview
│
├─ app_streamlit.py         # web demo serving the full pipeline
├─ report_pdf.py            # PDF report generator for demo results
│
├─ retfound_seg/            # ① disc/cup segmentation + C/D ratio
│  ├─ dataset.py            #   REFUGE segmentation dataset
│  ├─ model.py              #   RETFound encoder + segmentation decoder
│  ├─ metrics.py            #   Dice+CE loss / Dice metric
│  ├─ cdr.py                #   mask -> C/D ratio (vertical / area)
│  ├─ seg_train.py          #   training
│  ├─ infer.py              #   inference + C/D ratio (CSV / overlays)
│  ├─ make_overlays.py      #   prediction boundary overlays
│  └─ od_crop.py            #   fundus -> disc ROI crop
│
├─ glaucoma_cls/            # ② glaucoma classifier (RETFound + 11 concepts, linear head)
│  ├─ data.py               #   dataset pooling (REFUGE/ORIGA/G1020/GAMMA/GRAPE)
│  ├─ model.py              #   GlaucomaNet: RETFound encoder + concept concat + linear head
│  ├─ concepts.py           #   concept definitions/extraction (seg geometry + RNFL)
│  ├─ eyeon_cbm.py          #   concept-bottleneck extraction pipeline (used by the demo)
│  ├─ explain.py            #   concept_saliency / ViTGradCAM / attention_rollout / mc_dropout_ci
│  ├─ cv_runner.py          #   cross-validation sweeps
│  ├─ train.py              #   v1 training (GAMMA-only validation)
│  └─ train_v2.py           #   v2 training (pooled GAMMA+REFUGE+GRAPE, stratified 5-fold, OOF)
│
├─ feature_extraction/      # concept/embedding extraction feeding glaucoma_cls + bscan_gen
│  ├─ datasets_glaucoma.py  #   (image path, label) listing across datasets
│  ├─ extract_concepts.py   #   v1 concept cache (cbm_concepts.npz)
│  ├─ extract_concepts_v2.py#   v2 concept cache, 11 concepts (cbm_concepts_v2.npz)
│  ├─ extract_concepts_gamma_test.py # GAMMA test concept extraction
│  ├─ fit_oct_linear.py     #   v1 Stage A thickness PLS fit
│  └─ fit_rnfl_pls.py       #   v2 RNFL concept PLS fit (GRAPE ground truth)
│
└─ bscan_gen/                # ③ fundus -> macular OCT B-scan generation
   ├─ extract_slices.py     #   GAMMA .mhd/.raw volumes -> slice jpgs
   ├─ seg_ilmrpe.py         #   B-scan ILM/RPE layer segmentation (pseudo-labels)
   ├─ build_features.py     #   ILM/RPE -> thickness features + glaucoma correlation
   ├─ diffusion_bscan.py    #   conditional DDPM (single GPU), sketch -> OCT, used by the demo
   ├─ gen_from_handlabels.py#   generate from hand-labeled sketches
   ├─ fundus_to_oct_e2e.py  #   end-to-end: fundus -> predicted sketch -> OCT (512, best-of-5)
   ├─ ood_gate.py           #   rejects non-fundus inputs, used by the demo
   └─ utils.py              #   shared helpers (paths / embeddings / conditioning / flatten)
```

---

## Setup

```bash
# 1) install dependencies (match the torch build to your CUDA version)
pip install -r requirements.txt

# 2) place RETFound_cfp weights at:
#    <DATA_ROOT>/models/RETFound_cfp_weights.pth

# 3) sanity-check config/paths
python config.py
```

---

## ① retfound_seg — disc/cup segmentation + C/D ratio

REFUGE `Masks_Cropped` pixel values: `0=background, 1=disc rim, 2=cup` -> disc=(1∪2), cup=(2).

```bash
# train (best-val-Dice checkpoint saved to outputs/checkpoints/best.pth)
python -m retfound_seg.seg_train

# infer C/D ratio
#   local_config: INFER_MODE="split", INFER_SPLIT="test", INFER_SAVE_VIS=True/False
python -m retfound_seg.infer
#   single image: INFER_MODE="image", INFER_IMAGE="path/to/disc_crop.jpg"

# prediction boundary overlays
#   local_config: OVERLAY_SPLIT, OVERLAY_LIMIT (0=all), OVERLAY_OUT
python -m retfound_seg.make_overlays

# whole-fundus -> disc ROI crop (e.g. for GRAPE)
#   local_config: ODCROP_MARGIN, ODCROP_OUT_SIZE
python -m retfound_seg.od_crop
```

Relevant config groups: `DataConfig` (labels/input size), `ModelConfig` (backbone/decoder/freeze),
`TrainConfig` (epochs/batch/LR), `InferConfig` (vertical/area, postprocessing).

---

## ② glaucoma_cls — glaucoma classifier

```bash
# v2 (current): pooled GAMMA+REFUGE+GRAPE, stratified 5-fold, 11 concepts
python -m glaucoma_cls.train_v2

# v1 (reference): REFUGE+ORIGA+G1020 train, GAMMA-only validation, 9 concepts
# local_config: CLS_EPOCHS, CLS_BATCH_SIZE, CLS_LR_ENC, CLS_LR_HEAD,
#               CLS_FREEZE, CLS_ADD_GAMMA_TRAIN, CLS_PREDICT_ONLY
python -m glaucoma_cls.train
```

Output: `outputs/glaucoma_cls/glaucoma_v2_11concept_bestfold.pth` and out-of-fold predictions
(`oof_v2_11concept.npz`) used to recalibrate the decision thresholds
(`THR_SUSPECT`/`THR_HIGH` in `app_streamlit.py`).

**Recalibrate thresholds after every retrain** — probability scale shifts between runs, so
thresholds must be rescanned from the new OOF probabilities, not reused from a prior run.

---

## ③ bscan_gen — fundus -> macular OCT B-scan generation

**Two stages**: (Stage A) fundus embedding -> predicted macular thickness profile -> ILM/RPE
sketch; (Stage B) conditional diffusion renders that sketch into OCT texture.

```bash
# 0) extract slice jpgs from GAMMA volumes (.mhd/.raw), if not already extracted
python -m bscan_gen.extract_slices

# 1) train B-scan ILM/RPE layer segmentation & generate pseudo-labels
#    local_config: SEG_RUN_MODE="train"|"predict", SEG_EPOCHS,
#                  SEG_PREDICT_VOLUMES (None=all), SEG_SLICES_PER_VOL
python -m bscan_gen.seg_ilmrpe

# 2) extract thickness features + glaucoma correlation report
python -m bscan_gen.build_features

# 3) Stage A: fit fundus embedding -> thickness-profile PLS
python -m feature_extraction.fit_oct_linear

# 4) Stage B: train the conditional DDPM (single GPU)
#    local_config: DIFF_RUN_MODE="train"|"sample", DIFF_H/DIFF_W (resolution),
#                  DIFF_EPOCHS, DIFF_BS, DIFF_SAMPLE_N, FLATTEN_RPE
python -m bscan_gen.diffusion_bscan

# 5) generate
python -m bscan_gen.gen_from_handlabels   # hand-labeled sketch -> OCT
python -m bscan_gen.fundus_to_oct_e2e     # end-to-end fundus -> sketch -> OCT (512, best-of-5)

# OOD gate (optional)
#    local_config: GATE_CHECK_IMAGE=None (build gate) or an image path (check it)
python -m bscan_gen.ood_gate
```

**RPE flatten**: `FLATTEN_RPE=True` (default) removes the B-scan's scan-tilt artifact during
training/generation so layers are horizontally aligned. Checkpoints are saved separately as
`ddpm{H}_flat.pth`.

Relevant config: `OctTier1Config` (raw resolution 992x512, seg/diff resolutions, run modes).

---

## GAMMA pairing preview

```bash
# local_config: PREVIEW_N (number of cases), PREVIEW_CENTER (center slice index)
python gamma_preview.py
```

---

## Datasets

| Dataset | Used for | Notes |
|---|---|---|
| GAMMA | B-scan generation, hand-labeled ILM/RPE | 200 fundus-OCT pairs, macular 3x3mm (`D:/GAMMA`) |
| REFUGE | disc/cup segmentation, classification | train(400)/val(400) mask labels 0/1/2 |
| GRAPE | RNFL concept training, classification pool | all-glaucoma, real OCT RNFL thickness (n=244) |
| ORIGA / G1020 | not used in the current classifier | domain heterogeneity, see `../CLAUDE.md` §6 |

Project background, experiment history, and the full set of findings/rejected approaches are in
[`../CLAUDE.md`](../CLAUDE.md).
