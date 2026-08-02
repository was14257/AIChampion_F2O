# EYEON — 안저(fundus)에서 OCT 유래 정보 추출

컬러 안저사진(CFP) 한 장에서 OCT가 담고 있는 구조 정보를 끌어내는 프로젝트.
세 개의 파이프라인으로 구성된다.

1. **retfound_seg** — REFUGE 안저로 시신경유두(disc)/함몰부(cup)를 분할하고 **C/D ratio** 계산
2. **glaucoma_cls** — 외부 안저 데이터로 **녹내장 분류기**를 학습하고 GAMMA에 pseudo-label 생성
3. **bscan_gen** — 안저 → **황반 중앙 OCT B-scan 생성** (두께곡선 예측 + diffusion)

세 파이프라인을 한 번에 서빙하는 웹 데모는 [`app_streamlit.py`](app_streamlit.py) 하나뿐이다
(`streamlit run app_streamlit.py`) — fundus 한 장을 올리면 OCT 생성 + 녹내장 분류 +
concept 기반 설명가능성(concept_saliency 등)을 한 화면에서 보여준다.

> `f2o/`, `f2o-main/`는 별도 실험(slot-attention 버전)이라 이 파이프라인/설정 체계와 무관하다. 아래 설명은 그 두 폴더를 제외한다.

---

## 설정 방식 (중요)

**모든 실행 옵션은 [`local_config.py`](local_config.py) 한 파일에서 바꾼다. CLI 인자(`--flag`)는 쓰지 않는다.**
스크립트를 돌리기 전에 `local_config.py`에서 해당 값을 수정하고 실행하면 된다.

- [`local_config.py`](local_config.py) — 경로, 하이퍼파라미터, 실행 모드 등 **실제 값**
- [`config.py`](config.py) — `local_config` 값을 구조화한 `CFG` 제공 (`from config import CFG`)

경로는 로컬(Windows)/서버(`/home/tta`)를 자동 감지한다. GAMMA만 로컬에서 `D:/GAMMA` 별도.

---

## 폴더 구조

```
Code/Codes/
├─ local_config.py          # ★ 모든 설정 값 (경로/하이퍼파라미터/실행모드)
├─ config.py                # local_config → 구조화된 CFG
├─ requirements.txt
├─ gamma_preview.py         # GAMMA fundus↔중앙 B-scan 페어링 미리보기
│
├─ app_streamlit.py         # ★ 웹 데모 — 전 파이프라인 서빙 (fundus 업로드 → OCT 생성 + 녹내장 분류 + 설명가능성)
├─ report_pdf.py            # 데모 결과 PDF 리포트 생성
│
├─ retfound_seg/            # ① disc/cup 분할 + C/D ratio
│  ├─ dataset.py            #   REFUGE 데이터셋
│  ├─ model.py              #   RETFound 인코더 + 분할 디코더
│  ├─ metrics.py            #   Dice+CE 손실 / Dice 메트릭
│  ├─ cdr.py                #   마스크 → C/D ratio (vertical/area)
│  ├─ seg_train.py          #   학습
│  ├─ infer.py              #   추론 + C/D ratio (CSV/오버레이)
│  ├─ make_overlays.py      #   예측 경계선 오버레이 생성
│  └─ od_crop.py            #   안저 → disc ROI 크롭
│
├─ glaucoma_cls/            # ② 녹내장 분류기 (RETFound + concept 9종, 단일 Linear head)
│  ├─ data.py               #   REFUGE/ORIGA/G1020(학습) + GAMMA(검증/타겟)
│  ├─ model.py              #   GlaucomaNet: RETFound 인코더 + concept concat + Linear head
│  ├─ concepts.py           #   concept 9종(seg 기하 6 + OCT 3) 정의/추출
│  ├─ eyeon_cbm.py          #   concept bottleneck 추출 파이프라인 (데모가 사용)
│  ├─ explain.py            #   concept_saliency / ViTGradCAM / attention_rollout / mc_dropout_ci
│  ├─ cv_runner.py          #   5-fold CV
│  └─ train.py              #   학습 → GAMMA test pseudo-label
│
├─ feature_extraction/      #   concept/embedding 추출 등 전처리 스크립트 (glaucoma_cls가 참조하는 캐시 생성)
│  ├─ datasets_glaucoma.py  #   REFUGE/G1020/ORIGA/GAMMA (이미지경로,라벨) 목록 수집
│  ├─ extract_concepts.py   #   concept 9종 캐시 생성 (cbm_concepts.npz, data.py가 로드)
│  ├─ extract_concepts_gamma_test.py # GAMMA test 100장 concept 별도 추출 (train.py가 로드)
│  └─ fit_oct_linear.py     #   Stage A 두께 예측 PLS 최종 fit (oct_linear.pt, 데모가 로드)
│
└─ bscan_gen/              # ③ 안저 → 황반 중앙 OCT B-scan 생성
   ├─ extract_slices.py     #   GAMMA .mhd/.raw 볼륨 → 슬라이스 jpg
   ├─ seg_ilmrpe.py         #   B-scan ILM/RPE 층 분할 (pseudo-label 생성)
   ├─ build_features.py     #   ILM/RPE → 두께 feature + 녹내장 상관
   ├─ diffusion_bscan.py    #   조건부 DDPM (단일 GPU) — sketch→OCT 생성, 데모가 사용
   ├─ gen_from_handlabels.py#   손라벨 sketch → 생성 (ddim_sample 함수를 다른 스크립트가 재사용)
   ├─ fundus_to_oct_e2e.py # end-to-end: 안저 → 예측 sketch → OCT (512 해상도, best-of-5, 데모와 동일 로직)
   ├─ ood_gate.py           #   입력이 안저인지 게이트(비-안저 거절), 데모가 사용
   └─ utils.py              #   공용(경로/임베딩/조건맵/flatten 등), 데모가 사용
```

> Stage A(안저→두께곡선) 최종 fit은 `feature_extraction/fit_oct_linear.py`에 있다.
> 탐색용 스크립트(`stageA_improve.py`, `stageA_nasal_temporal_search.py`)와 미사용
> 구버전(256 해상도판, 멀티 GPU `diffusion_ddp.py`, best-of-1판)은 2026-08-02
> 정리에서 삭제됨 — 남은 `fundus_to_oct_e2e.py`가 원래 이름
> `fundus_to_oct_e2e_512_best5.py`에서 리네임된 최종본(CLAUDE.md 4절 참고).

---

## 설치

```bash
# 1) 의존성 (CUDA 버전에 맞는 torch 설치)
pip install -r requirements.txt

# 2) RETFound_cfp 가중치를 아래 경로에 둔다
#    <DATA_ROOT>/models/RETFound_cfp_weights.pth

# 3) 설정/경로 점검
python config.py
```

---

## ① retfound_seg — disc/cup 분할 + C/D ratio

REFUGE `Masks_Cropped` 픽셀값: `0=배경, 1=disc rim, 2=cup` → disc=(1∪2), cup=(2).

```bash
# 학습 (val mean Dice 최고 모델을 outputs/checkpoints/best.pth 저장)
python -m retfound_seg.seg_train

# C/D ratio 추론
#   local_config: INFER_MODE="split", INFER_SPLIT="test", INFER_SAVE_VIS=True/False
python -m retfound_seg.infer
#   단일 이미지: INFER_MODE="image", INFER_IMAGE="path/to/disc_crop.jpg"

# 예측 경계선 오버레이 생성
#   local_config: OVERLAY_SPLIT, OVERLAY_LIMIT(0=전체), OVERLAY_OUT
python -m retfound_seg.make_overlays

# 전체 안저 → disc ROI 크롭 (GRAPE 등)
#   local_config: ODCROP_MARGIN, ODCROP_OUT_SIZE
python -m retfound_seg.od_crop
```

관련 설정: `DataConfig`(라벨/입력크기), `ModelConfig`(백본/디코더/동결),
`TrainConfig`(epoch/batch/LR), `InferConfig`(vertical/area, 후처리).

---

## ② glaucoma_cls — 녹내장 분류기

라벨 있는 외부 안저(REFUGE train + ORIGA + G1020)로 학습 → GAMMA train으로 검증(도메인 전이) → GAMMA test에 pseudo-label.

```bash
# local_config: CLS_EPOCHS, CLS_BATCH_SIZE, CLS_LR_ENC, CLS_LR_HEAD,
#               CLS_FREEZE(인코더 동결), CLS_ADD_GAMMA_TRAIN, CLS_PREDICT_ONLY
python -m glaucoma_cls.train
```

출력: `outputs/glaucoma_cls/best.pth`, `gamma_test_pseudolabels.csv` (Youden's J로 threshold 보정).

---

## ③ bscan_gen — 안저 → 황반 중앙 OCT B-scan 생성

**2단계**: (Stage A) 안저 임베딩 → 황반 두께곡선 예측으로 ILM/RPE sketch 구성 →
(Stage B) 조건부 diffusion으로 sketch를 실제 OCT 텍스처로 변환.

```bash
# 0) GAMMA 볼륨(.mhd/.raw)에서 슬라이스 jpg 추출 (슬라이스가 없을 때)
python -m bscan_gen.extract_slices

# 1) B-scan ILM/RPE 층 분할 학습 & pseudo-label 생성
#    local_config: SEG_RUN_MODE="train"|"predict", SEG_EPOCHS,
#                  SEG_PREDICT_VOLUMES(None=전체), SEG_SLICES_PER_VOL
python -m bscan_gen.seg_ilmrpe

# 2) 두께 feature 추출 + 녹내장 상관 리포트
python -m bscan_gen.build_features

# 3) Stage A: 안저 임베딩 → 두께곡선 PLS 최종 fit
python -m feature_extraction.fit_oct_linear

# 4) Stage B: 조건부 DDPM 학습 (단일 GPU)
#    local_config: DIFF_RUN_MODE="train"|"sample", DIFF_H/DIFF_W(해상도),
#                  DIFF_EPOCHS, DIFF_BS, DIFF_SAMPLE_N, FLATTEN_RPE
python -m bscan_gen.diffusion_bscan

# 5) 생성
python -m bscan_gen.gen_from_handlabels          # 손라벨 sketch → OCT
python -m bscan_gen.fundus_to_oct_e2e  # 안저 → 예측 sketch → OCT (end-to-end, 512, best-of-5, 데모와 동일)

# 입력이 안저인지 게이트 (선택)
#    local_config: GATE_CHECK_IMAGE=None(게이트 구축) 또는 이미지 경로(검사)
python -m bscan_gen.ood_gate
```

**RPE flatten**: `FLATTEN_RPE=True`(기본)면 학습/생성 시 B-scan의 스캔 기울기 아티팩트를
제거해 층을 수평으로 정렬한다. 체크포인트는 `ddpm{H}_flat.pth`로 분리 저장된다.

관련 설정: `OctTier1Config`(원본해상도 992×512, seg/diff 해상도, 실행모드 등).

---

## GAMMA 페어링 미리보기

```bash
# local_config: PREVIEW_N(케이스 수), PREVIEW_CENTER(중앙 슬라이스 index)
python gamma_preview.py
```

---

## 데이터

| 데이터셋 | 용도 | 비고 |
|---|---|---|
| REFUGE | disc/cup 분할, 분류 | 마스크 라벨 0/1/2 |
| ORIGA / G1020 | 분류 학습 | 녹내장 라벨 |
| GAMMA | B-scan 생성 | 안저+OCT 볼륨 200쌍, 황반 중심 (`D:/GAMMA`) |
| GRAPE | disc 크롭 대상 | 녹내장 전용 |

프로젝트 배경/실험 요약/진행 히스토리는 [`../CLAUDE.md`](../CLAUDE.md) 참고.
