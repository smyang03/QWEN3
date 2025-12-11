# 빠른 시작 가이드

## 체크리스트

### 1단계: 환경 설정 ✓
```bash
# 1. Python 환경 확인
python --version  # Python 3.8+ 필요

# 2. 가상환경 생성 (선택사항)
conda create -n label_verifier python=3.10 -y
conda activate label_verifier

# 3. 의존성 설치
cd /path/to/yolo_label_verifier
pip install -r requirements.txt

# 4. GPU 확인 (CUDA 필요)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 2단계: 데이터 준비 ✓
```bash
# 1. 입력 폴더 확인
ls input/images/  # 이미지 파일들이 있어야 함
ls input/labels/  # 대응하는 .txt 라벨 파일들

# 2. 라벨 형식 확인
cat input/labels/example.txt
# 예상 출력: 
# 0 0.5 0.5 0.3 0.4
# 1 0.2 0.3 0.15 0.25
```

### 3단계: 설정 확인 ✓
```bash
# config.yaml 열어서 클래스 매핑 확인
cat config.yaml | grep -A 10 "classes:"

# 출력 예시:
# classes:
#   0: "person"
#   1: "car"
#   2: "helmet"
#   3: "motorcycle"

# 필요시 수정
nano config.yaml  # 또는 vim, vi, code 등
```

### 4단계: 실행 ✓
```bash
# 검증 실행
python verifier.py

# 실행 과정:
# [1] 모델 목록 표시
# [2] 모델 번호 선택 (예: 3)
# [3] 확인 (y/n): y
# [4] 모델 다운로드 (최초 1회)
# [5] 검증 시작 (진행률 표시)
# [6] 결과 저장
```

### 5단계: 결과 확인 ✓
```bash
# 출력 폴더 확인
tree output/
# output/
# ├── correct/
# │   ├── images/
# │   └── labels/
# ├── mislabeled/
# │   ├── images/
# │   └── labels/
# ├── uncertain/
# │   ├── images/
# │   └── labels/
# ├── verification_report.json
# └── summary.txt

# 요약 확인
cat output/summary.txt

# 상세 리포트
cat output/verification_report.json | jq .statistics
```

## 일반적인 문제 해결

### ❌ ImportError: No module named 'transformers'
```bash
pip install -r requirements.txt
```

### ❌ CUDA out of memory
```yaml
# config.yaml 수정
verification:
  max_image_size: 640  # 기본값: 1280
```
또는 더 작은 모델 선택 (2B 또는 4B)

### ❌ No images found!
```bash
# 이미지 경로 확인
ls input/images/*.jpg  # 또는 .png

# 경로가 다르면 config.yaml 수정
paths:
  input_images: "your/actual/path/images"
```

### ❌ Label not found for XXX.jpg
```bash
# 라벨 파일 이름이 이미지와 일치하는지 확인
# 예: image_001.jpg -> image_001.txt

# 파일 목록 비교
ls input/images/ | cut -d. -f1 | sort > images.txt
ls input/labels/ | cut -d. -f1 | sort > labels.txt
diff images.txt labels.txt
```

## 첫 실행 예상 시간

| 항목 | 시간 |
|------|------|
| 환경 설정 | 5-10분 |
| 모델 다운로드 (8B) | 10-20분 (네트워크 속도에 따라) |
| 100개 이미지 검증 (평균 3박스) | 5분 (8B 모델, RTX 4090 기준) |
| **총 소요 시간** | **약 20-35분** |

## 추천 설정 (첫 실행)

```yaml
# config.yaml - 추천 설정

# 처음엔 작은 모델로 테스트
# 모델 선택: 2 (Qwen3-VL-4B-Instruct)

verification:
  mode: "single"
  confidence_threshold: 0.7
  create_uncertain_folder: true
  crop_padding: 10
  max_image_size: 1280

processing:
  show_progress: true
  device: "auto"
  empty_cache_frequency: 50  # 메모리 부족시 줄이기
```

## 다음 단계

1. ✅ 설정 커스터마이징
2. ✅ Uncertain 케이스 검토
3. ✅ 더 큰 모델로 재검증 (필요시)
4. ✅ 통계 분석
5. ✅ 데이터셋 개선

## 유용한 명령어

```bash
# 통계 빠르게 확인
cat output/summary.txt

# 오라벨된 이미지만 보기
ls output/mislabeled/images/

# 클래스별 정확도 확인
cat output/verification_report.json | jq '.statistics.per_class'

# 검증 시간 확인
cat output/verification_report.json | jq '.duration_seconds'

# 특정 클래스의 오라벨 비율
cat output/verification_report.json | jq '.statistics.per_class.person'
```

## 문제 발생시

1. log 파일 확인 (콘솔 출력)
2. config.yaml 설정 재확인
3. GPU 메모리 확인: `nvidia-smi`
4. 작은 샘플로 먼저 테스트 (10개 이미지)
