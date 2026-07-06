# RETFound vs U-Net: OD/OC Segmentation 비교

민준이 요청 — RETFound 인코더 기반 OD/OC segmentation을 구현하고, 생 U-Net과 성능을 비교.
동양인 코호트 데이터셋 기준 (REFUGE2 / GAMMA / GRAPE, 전부 중국 안과 데이터).

> ✅ 업데이트: 이제 더미 텐서 검증을 넘어서, **실제 안저 이미지로 학습 루프 전체(U-Net, RETFound 둘 다)가
> 실제로 도는 것까지 로컬(MPS)에서 확인함** — 데이터 로딩 → augmentation → forward/backward →
> Dice 계산 → 체크포인트 저장 → `compare.py` 로 비교표 출력까지 전 과정 통과.
> (RETFound 쪽은 timm ViT-L 아키텍처와 키가 100% 같은 체크포인트로 로드 경로만 검증했고,
> 공식 RETFound 사전학습 가중치·실제 REFUGE2/GAMMA/GRAPE 정답 마스크로 돌렸을 때
> 나오는 실제 Dice 수치까지 검증한 건 아니야 — 그건 진짜 데이터 받은 다음에 확인해야 함.)

## 1. 환경 세팅

```bash
conda create -n f2o_seg python=3.10 -y
conda activate f2o_seg
pip install -r requirements.txt
# GPU 서버라면 CUDA 버전 맞는 torch로 따로 설치 권장:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ **꼭 Python 3.10 이상 써야 함.** 코드에 `str | None` 같은 3.10+ 타입 힌트 문법이 있어서
> 시스템 기본 `python3`(맥은 보통 3.9)로 그냥 돌리면 `TypeError: unsupported operand type(s) for |`
> 로 바로 죽어. `conda activate f2o_seg` 하고 나서 `python --version`으로 3.10인지 먼저 확인해줘
> (이 env 안에서는 `python3`가 아니라 `python` 커맨드를 써야 할 수도 있음 — conda env에 `python3` 심볼릭 링크가
> 없는 경우가 있어서, `python3 train.py ...`가 아니라 `python train.py ...`로 실행해).

## 2. RETFound 사전학습 가중치 받기

공식 저장소: https://github.com/rmaphoh/RETFound (구 rmaphoh/RETFound_MAE)
- **ViT-Large, Colour fundus image** 가중치 다운로드 (README의 "Colour fundus image" 링크)
- 지금은 HuggingFace에도 올라가 있어서 `huggingface_hub`로도 받을 수 있음

받은 체크포인트 경로를 `--retfound_ckpt`에 넣어주면 돼. 우리 `models/retfound_backbone.py`가
`state_dict`를 `strict=False`로 로드하면서 missing/unexpected 키를 출력해주니까,
실제로 돌려보고 로그에 뜨는 키 목록 캡처해서 보내주면 필요시 키 매핑 추가해줄게
(RETFound 원 저장소의 `models_vit.py`가 timm ViT랑 100% 동일 키는 아닐 수 있음).

## 3. 데이터 폴더 구조

```
REFUGE2/
  images/
    train/  xxx.jpg ...
    val/    xxx.jpg ...
    test/   xxx.jpg ...
  masks/
    train/  xxx.bmp ...   # 파일명(stem)이 이미지랑 동일해야 함
    val/    ...
    test/   ...
```

마스크는 REFUGE 관례 그레이스케일 3-level (0=cup, 128=disc-ring, 255=background).

## 3-1. 원본 데이터셋 -> 위 구조로 변환 (`prepare_dataset.py`)

REFUGE2 / GAMMA / GRAPE는 공식 배포 폴더 구조나 마스크 픽셀값 관례가 서로 조금씩 달라서
(예: GRAPE의 "Annotated Images"는 이미지 위에 초록/빨강 윤곽선이 이미 합성된 시각화용이라
정답 마스크로 못 쓰고, 실제 정답은 별도 "Segmentation masks" 배포본에 있음),
`prepare_dataset.py`가 아래를 자동으로 해줘:

- 이미지 폴더 / 마스크 폴더를 각각 재귀로 스캔해서 파일명(stem) 기준으로 짝을 맞춤
  (마스크 파일명에 `_mask`, `_gt`, `_seg` 같은 접미사가 붙어있어도 자동으로 벗겨서 재시도함)
- 마스크 그레이스케일 값(255/128/0 인지, 0/1/2 인지 등)을 샘플링해서 자동으로 판별하고
  REFUGE 관례(255=bg, 128=disc-ring, 0=cup)로 재인코딩해서 저장 — 그래서 `data/refuge_dataset.py`는
  항상 REFUGE 관례만 읽으면 되고 데이터셋별 분기 코드가 필요 없음
- OD/OC가 마스크 파일 하나에 안 들어있고 따로 배포되는 경우(`--od_mask_dir` / `--oc_mask_dir`)도 지원
- 공식 train/val/test가 이미 나뉜 배포본(REFUGE2, GAMMA)은 폴더별로 `--split`을 지정해서 3번 실행,
  공식 split이 없는 코호트(GRAPE)는 `--split_ratios`로 한 번에 무작위 분할

```bash
# REFUGE2 (보통 Training/Validation/Test가 원본 zip에서부터 폴더로 나뉘어 옴 -> 3번 실행)
python prepare_dataset.py --images_dir /path/to/REFUGE-Training400 \
    --masks_dir /path/to/Annotation-Training400 --out_root ./REFUGE2 --split train
python prepare_dataset.py --images_dir /path/to/REFUGE-Validation400 \
    --masks_dir /path/to/REFUGE-Validation400-GT --out_root ./REFUGE2 --split val
python prepare_dataset.py --images_dir /path/to/REFUGE-Test400 \
    --masks_dir /path/to/REFUGE-Test400-GT --out_root ./REFUGE2 --split test

# GAMMA (fundus 이미지 + OD/OC 마스크, 배포 버전에 따라 train/val 폴더가 나뉘어 있으면 REFUGE2처럼 --split으로,
# 안 나뉘어 있으면 GRAPE처럼 --split_ratios로)
python prepare_dataset.py --images_dir /path/to/GAMMA_fundus --masks_dir /path/to/GAMMA_disc_cup_mask \
    --out_root ./GAMMA --split_ratios 0.7,0.15,0.15 --seed 42

# GRAPE (CFPs = 원본 이미지, Segmentation masks = 실제 정답 마스크 — Annotated Images 아님!)
python prepare_dataset.py --images_dir /path/to/GRAPE_CFPs --masks_dir /path/to/GRAPE_Segmentation_masks \
    --out_root ./GRAPE --split_ratios 0.7,0.15,0.15 --seed 42
```

먼저 `--dry_run`을 붙여서 몇 장이 매칭되고 마스크 값이 어떻게 해석됐는지 확인해보고,
이상하면 `--mask_map "255:0,128:1,0:2"` 처럼 직접 지정해줘. 변환이 끝나면 `--out_root`로
지정한 폴더를 그대로 `train.py --data_root`에 넘기면 됨 — 데이터셋이 REFUGE2든 GAMMA든
GRAPE든 이후 명령어는 완전히 동일해.

> 폴더 안의 `grape_annotated/`, `grape_cfps.zip`, `grape_annotated.zip`은 참고용으로 남겨둔 거고
> (그마저도 zip 2개는 다운로드가 깨져서 0바이트임), git에는 올리지 않음 (`.gitignore` 참고).
> `grape_annotated/`는 시각화 이미지라 그대로는 학습에 못 쓰고, 실제 GRAPE 정답 마스크를
> 받으면 위 `prepare_dataset.py` 커맨드로 변환해서 쓰면 됨.

## 4. 학습

```bash
# U-Net baseline
python train.py --model unet --data_root ./REFUGE2 \
    --epochs 50 --batch_size 8 --output_dir runs/unet

# RETFound (encoder frozen, decoder만 학습) — 우선 이걸로 먼저 돌려봐
python train.py --model retfound --data_root ./REFUGE2 \
    --retfound_ckpt ./RETFound_cfp_weights.pth \
    --epochs 50 --batch_size 8 --output_dir runs/retfound_frozen

# (선택) RETFound encoder까지 fine-tune — lr 낮게, 더 오래 걸림
python train.py --model retfound --data_root ./REFUGE2 \
    --retfound_ckpt ./RETFound_cfp_weights.pth --no_freeze_encoder \
    --lr 1e-4 --epochs 50 --output_dir runs/retfound_finetuned
```

각 run의 `runs/<name>/log.csv`에 epoch별 loss/Dice가 쌓이고,
best mean Dice 기준 체크포인트가 `runs/<name>/best.pt`로 저장돼.

## 5. 비교

```bash
python compare.py --data_root ./REFUGE2 \
    --unet_ckpt runs/unet/best.pt \
    --retfound_ckpt runs/retfound_frozen/best.pt \
    --retfound_weights ./RETFound_cfp_weights.pth
```

Dice(OD) / Dice(OC) / 평균 Dice / 파라미터 수를 표로 출력해줌.
OC가 저대비·경계 모호해서 어려운 태스크라 RETFound 우위가 여기서 더 크게
나올 가능성이 높다는 게 우리 예상 — 실제로 그렇게 나오면 논문에 넣기 좋은 포인트.

## 5-1. 로컬 맥북에서 돌릴 때 (서버 대신)

학교 GPU 서버가 캠퍼스 IP만 허용해서 접속 안 되는 상황이면, 일단 로컬 맥북으로 진행해도 코드는
그대로 동작해 — `train.py`/`compare.py`가 `torch.backends.mps.is_available()`로 Apple Silicon GPU(MPS)를
자동 감지해서 씀.

다만 RETFound는 ViT-Large(3억+ 파라미터)라 노트북에서 돌리려면 이렇게 하는 걸 추천해:

- **encoder는 무조건 freeze 상태로 시작** (`--no_freeze_encoder` 플래그 쓰지 말기). encoder fine-tuning은
  gradient가 3억 파라미터 전체에 걸리기 때문에 메모리가 훨씬 많이 필요하고, 노트북에서는 비현실적임
  (frozen이면 `models/retfound_backbone.py`가 자동으로 `torch.no_grad()`로 감싸서 메모리 아껴줌)
- **batch_size 작게** (`--batch_size 2` 나 `4`부터 시작, 안 터지면 조금씩 올리기)
- **먼저 U-Net으로 전체 파이프라인부터 검증** — U-Net은 가벼워서(3천만 파라미터) 데이터 로더·마스크
  변환·loss 계산이 다 제대로 도는지 빠르게 확인할 수 있음. U-Net이 문제없이 돌면 그다음 RETFound로 넘어가기
- 그래도 메모리 부족(`zsh: killed` 또는 `MPS backend out of memory`) 뜨면 `--img_size`를 224보다
  낮추는 것보다는(RETFound가 224 고정 학습이라 해상도 바꾸면 위치 임베딩 보간이 들어가서 성능 저하 가능)
  batch_size를 1까지 낮추는 걸 먼저 시도해봐

```bash
# 로컬 맥북 - U-Net 먼저 (파이프라인 검증용)
python train.py --model unet --data_root ./REFUGE2 \
    --epochs 20 --batch_size 8 --output_dir runs/unet

# 로컬 맥북 - RETFound (encoder frozen, batch_size 작게)
python train.py --model retfound --data_root ./REFUGE2 \
    --retfound_ckpt ./RETFound_cfp_weights.pth \
    --epochs 20 --batch_size 2 --output_dir runs/retfound_frozen
```

## 6. 구조 메모

- `models/retfound_backbone.py` : ViT-Large/16 (timm) + RETFound 체크포인트 로더
  + 위치 임베딩 보간 (입력 해상도 다를 때 대비)
- `models/segmentation_models.py` :
  - `RETFoundSegmenter` = frozen/fine-tune 가능한 RETFound 인코더 + Segmenter 스타일
    mask-transformer decoder (class mask token과 patch token 내적으로 마스크 생성)
  - `UNetBaseline` = 처음부터 학습하는 표준 U-Net
- `utils/losses.py` : Dice+CE 결합 loss, OD/OC Dice 평가 함수
  (OD는 "OD-ring ∪ OC"로 재구성해서 계산 — REFUGE 관례와 동일)
- `data/refuge_dataset.py` : 이미지/마스크 페어 로더 + 기본 augmentation
  (flip, ±15도 회전 — FunduSegmenter 논문에서도 "복잡한 augmentation보다
  기본 spatial augmentation이 RETFound fine-tuning에 더 잘 먹힌다"고 보고함)
- `prepare_dataset.py` : REFUGE2/GAMMA/GRAPE 등 원본 배포 구조 -> 위 표준 구조 변환 (3-1 참고)

## 다음으로 확장 가능한 것

- risk scoring 이어붙이기: 여기서 나온 OD/OC mask로 CDR 계산 → f2o Option 3
  파이프라인의 Risk MLP 입력으로 연결 (민준이가 카톡에서 물어봤던 부분)
- GRAPE ↔ REFUGE2 cross-dataset 평가로 도메인 일반화 비교 추가
