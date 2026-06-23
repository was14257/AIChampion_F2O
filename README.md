# RETFound 기반 C/D ratio 파이프라인

REFUGE 안저 데이터로 **RETFound(ViT-L/16) 인코더 + 세그멘테이션 디코더**를 학습해
시신경유두(disc)/함몰부(cup)를 분할하고, 거기서 **C/D ratio**를 계산한다.

## 구조

```
Code/
├─ local_config.py          # ★ 모든 설정(경로/하이퍼파라미터)을 클래스로 정의. CFG 로 import
├─ train.py                 # 학습 (best.pth 저장)
├─ infer.py                 # 추론 + C/D ratio 계산 (CSV/오버레이 저장)
├─ requirements.txt
├─ RETFound_DOWNLOAD.md     # RETFound 가중치 받는 법
└─ retfound_seg/
   ├─ dataset.py            # REFUGE disc/cup 데이터셋 (ROI 크롭본 사용)
   ├─ model.py              # RETFound 인코더 로딩 + 디코더
   ├─ cdr.py                # 마스크 → C/D ratio (vertical/area)
   └─ metrics.py            # Dice+CE 손실, Dice 메트릭
```

## 데이터 라벨 규약

REFUGE `Masks_Cropped` PNG 픽셀값: `0=배경, 1=disc rim, 2=cup`
→ disc 전체 = (1∪2), cup = (2). ROI 크롭본은 디스크 중심 정사각이라 분할에 적합.

## 사용 순서

```bash
# 1) 의존성 (CUDA 버전에 맞는 torch 설치)
pip install -r requirements.txt

# 2) RETFound_cfp 가중치 받기  → Data/models/RETFound_cfp_weights.pth
#    (RETFound_DOWNLOAD.md 참고)

# 3) 설정 점검 (경로/가중치 존재 여부 확인)
python local_config.py

# 4) 학습  (val mean Dice 최고 모델을 outputs/checkpoints/best.pth 로 저장)
python train.py

# 5) C/D ratio 추론
python infer.py --split test                 # split 전체 → outputs/predictions/cdr_test.csv
python infer.py --image path/to/disc_crop.jpg  # 단일 ROI 크롭 이미지
```

## 설정 바꾸기

전부 [local_config.py](local_config.py) 한 곳에서:
- 경로 → `PathConfig`
- 라벨/입력크기/정규화 → `DataConfig`
- 백본/디코더/특징층/인코더동결 → `ModelConfig`
- epoch/batch/LR/손실가중 → `TrainConfig`
- C/D ratio 종류(vertical/area)/후처리 → `InferConfig`
- device(자동 CUDA 감지) → `RuntimeConfig`

## 참고

- REFUGE 에는 C/D ratio **정답 수치가 없어서**, `infer.py` 는 GT 마스크에서
  계산한 C/D ratio 를 기준으로 예측값과의 MAE/상관계수를 보고한다.
- C/D ratio 수치 정답이 필요하면 ORIGA(`ExpCDR`)로 교차검증할 수 있다.
- 새 이미지(크롭 안 된 전체 안저)는 먼저 디스크 ROI 크롭이 필요하다
  (현재 파이프라인은 데이터셋의 크롭본을 전제로 함).
