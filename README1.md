# YOLO Label Verifier using Qwen3-VL

SAM3로 생성된 YOLO 라벨을 Qwen3-VL 모델로 자동 검증하는 시스템

## 🎯 주요 기능

- ✅ **모델 선택**: 5가지 Qwen3-VL 모델 중 선택 가능
- ✅ **자동 캐싱**: 모델을 한 번만 다운로드
- ✅ **박스 크롭**: 박스 영역만 추출하여 검증
- ✅ **배치/단일 처리**: 설정으로 선택 가능
- ✅ **3단계 분류**: correct / mislabeled / uncertain
- ✅ **상세 리포트**: JSON + 텍스트 요약
- ✅ **진행률 표시**: tqdm으로 실시간 진행 상황
- ✅ **클래스별 통계**: 클래스마다 정확도 측정

## 📁 디렉토리 구조

```
yolo_label_verifier/
├── config.yaml           # 설정 파일
├── verifier.py          # 메인 스크립트
├── requirements.txt     # 의존성
├── models/              # 모델 캐시 (자동 생성)
├── input/               # 입력 데이터
│   ├── images/          # 이미지 파일
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/          # YOLO 라벨 파일
│       ├── img_001.txt
│       └── ...
└── output/              # 결과 (자동 생성)
    ├── correct/         # 정확한 라벨
    │   ├── images/
    │   └── labels/
    ├── mislabeled/      # 잘못된 라벨
    │   ├── images/
    │   └── labels/
    ├── uncertain/       # 애매한 경우
    │   ├── images/
    │   └── labels/
    ├── verification_report.json
    └── summary.txt
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
conda create -n label_verifier python=3.10 -y
conda activate label_verifier

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 입력 폴더에 데이터 복사
cp -r /path/to/your/images/* ./input/images/
cp -r /path/to/your/labels/* ./input/labels/
```

**YOLO 라벨 형식 (txt):**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.25
```
- `class_id x_center y_center width height` (normalized)

### 3. 설정 수정 (선택사항)

`config.yaml` 파일을 열어 설정 수정:

```yaml
# 클래스 매핑 수정
classes:
  0: "person"
  1: "car"
  2: "helmet"
  3: "motorcycle"

# 검증 모드 선택
verification:
  mode: "single"  # "single" 또는 "batch"
  crop_padding: 10
  confidence_threshold: 0.7
```

### 4. 실행

```bash
python verifier.py
```

**첫 실행시:**
1. 사용 가능한 모델 목록이 표시됩니다
2. 숫자를 입력하여 모델 선택 (추천: 3번 - Qwen3-VL-8B)
3. 모델 다운로드 (최초 1회만, 시간 소요)
4. 자동으로 검증 시작

**이후 실행시:**
- 캐시된 모델을 자동으로 로드하여 즉시 시작

## ⚙️ 주요 설정 옵션

### 모델 설정
```yaml
model:
  cache_dir: "./models"  # 모델 저장 위치
```

### 검증 설정
```yaml
verification:
  mode: "single"         # single: 박스 하나씩, batch: 여러 박스 동시
  batch_size: 4          # batch 모드일 때 배치 크기
  crop_method: "crop"    # crop: 박스만, full_image: 전체 이미지
  
  confidence_threshold: 0.7    # 신뢰도 임계값
  create_uncertain_folder: true  # 애매한 경우 별도 저장
  
  crop_padding: 10       # 크롭시 패딩 (픽셀)
  crop_min_size: 50      # 최소 크롭 크기
  max_image_size: 1280   # 모델 입력 최대 크기
```

### 처리 옵션
```yaml
processing:
  show_progress: true    # 진행률 표시
  device: "auto"         # "auto", "cuda", "cpu"
  empty_cache_frequency: 100  # N개마다 캐시 비우기
  max_retries: 3         # 실패시 재시도 횟수
```

## 📊 결과 파일

### 1. verification_report.json
```json
{
  "timestamp": "2025-12-10T15:30:00",
  "duration_seconds": 450.5,
  "statistics": {
    "total_images": 1000,
    "total_boxes": 3500,
    "correct": 3200,
    "mislabeled": 250,
    "uncertain": 50,
    "per_class": {
      "person": {
        "correct": 1500,
        "mislabeled": 100,
        "uncertain": 20
      },
      ...
    }
  },
  "detailed_results": [...]
}
```

### 2. summary.txt
```
================================================================================
YOLO Label Verification Summary
================================================================================

Total Images: 1000
Total Boxes: 3500
Duration: 450.50 seconds

Overall Results:
  Correct: 3200 (91.4%)
  Mislabeled: 250 (7.1%)
  Uncertain: 50 (1.4%)

Per-Class Results:

  person:
    Correct: 1500 (92.6%)
    Mislabeled: 100 (6.2%)
    Uncertain: 20 (1.2%)
  ...
```

## 🎛️ 사용 예시

### 예시 1: 빠른 검증 (4B 모델)
```yaml
# config.yaml
verification:
  mode: "single"
  confidence_threshold: 0.7
```
```bash
python verifier.py
# 모델 선택: 2 (Qwen3-VL-4B-Instruct)
```
- 속도: ~0.5초/박스
- VRAM: ~8GB
- 용도: 대량 데이터 1차 스크리닝

### 예시 2: 정확한 검증 (8B 모델)
```yaml
# config.yaml
verification:
  mode: "single"
  confidence_threshold: 0.8
  create_uncertain_folder: true
```
```bash
python verifier.py
# 모델 선택: 3 (Qwen3-VL-8B-Instruct)
```
- 속도: ~1초/박스
- VRAM: ~16GB
- 용도: 최종 검증

### 예시 3: 배치 처리 (테스트 중)
```yaml
# config.yaml
verification:
  mode: "batch"
  batch_size: 4
```
- 여러 박스를 동시에 처리 (구현 예정)

## 🔧 트러블슈팅

### Q: CUDA out of memory 에러
**A:** 더 작은 모델 선택 또는 설정 수정:
```yaml
verification:
  max_image_size: 640  # 이미지 크기 줄이기
```

### Q: 모델 다운로드가 느림
**A:** cache_dir를 SSD 경로로 변경:
```yaml
model:
  cache_dir: "/fast/ssd/path/models"
```

### Q: 검증이 너무 느림
**A:** 4B 모델 사용 또는 GPU 확인:
```bash
# GPU 사용 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: 응답 파싱 오류가 많음
**A:** confidence_threshold를 낮추거나 uncertain 폴더 활성화:
```yaml
verification:
  confidence_threshold: 0.6
  create_uncertain_folder: true
```

## 📈 성능 가이드

### 모델별 예상 처리 속도 (RTX 4090 기준)

| 모델 | VRAM | 속도/박스 | 1000박스 소요시간 |
|------|------|-----------|-------------------|
| 2B | ~4GB | ~0.3초 | ~5분 |
| 4B | ~8GB | ~0.5초 | ~8분 |
| 8B | ~16GB | ~1.0초 | ~17분 |
| 32B | ~64GB | ~3.0초 | ~50분 |
| 30B-A3B | ~60GB | ~2.0초 | ~33분 |

*실제 속도는 이미지 크기, GPU 등에 따라 다름*

## 🎯 추천 워크플로우

### 1단계: 빠른 스크리닝 (4B)
```bash
# 4B 모델로 전체 데이터 빠르게 검증
python verifier.py  # 모델: 4B
```

### 2단계: 의심 케이스 재검증 (8B)
```bash
# mislabeled와 uncertain을 input으로 다시 검증
cp -r output/mislabeled/images/* input/images/
cp -r output/mislabeled/labels/* input/labels/
python verifier.py  # 모델: 8B
```

### 3단계: 최종 확인
```bash
# 남은 mislabeled만 수동 검토
```

## 📝 라이센스

- 코드: Apache 2.0
- Qwen3-VL 모델: Apache 2.0

## 🤝 기여

버그 리포트 및 기능 제안 환영합니다!

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.
