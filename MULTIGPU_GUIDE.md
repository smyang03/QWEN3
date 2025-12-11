# Multi-GPU 실행 가이드

YOLO Label Verifier를 여러 GPU에서 병렬로 실행하는 방법입니다.

---

## 📁 제공되는 스크립트

### 1. **run_multigpu_simple.sh** (추천 ⭐)
- **SAM3 스타일**의 간단한 버전
- 백그라운드 프로세스로 병렬 실행
- 가장 직관적이고 이해하기 쉬움

### 2. **run_multigpu.sh**
- 고급 기능 포함 버전
- 상세한 로깅 및 에러 처리
- 더 많은 옵션 제공

### 3. **multigpu_wrapper.py**
- 크로스 플랫폼 Python 래퍼
- Windows/Linux 모두 지원
- `multiprocessing` 사용

### 4. **run_multigpu.bat**
- Windows 전용 배치 파일
- `multigpu_wrapper.py` 호출

---

## 🚀 빠른 시작 (추천)

### Linux/WSL

```bash
# 실행 권한 부여 (최초 1회)
chmod +x run_multigpu_simple.sh

# 실행 (GPU 개수 자동 감지)
./run_multigpu_simple.sh

# 또는 GPU 개수 지정
./run_multigpu_simple.sh 4
```

### Windows

```batch
REM Python wrapper 직접 실행
python multigpu_wrapper.py

REM 또는 배치 파일 사용
run_multigpu.bat
```

---

## 📊 동작 원리

### 1. **이미지 분할**
```
Total: 1000 images
GPU 0: 250 images (0-249)
GPU 1: 250 images (250-499)
GPU 2: 250 images (500-749)
GPU 3: 250 images (750-999)
```

### 2. **병렬 실행**
```
[GPU 0] Processing split_00...
[GPU 1] Processing split_01...
[GPU 2] Processing split_02...
[GPU 3] Processing split_03...
```

### 3. **결과 통합**
```
output/
├── correct/
├── mislabeled/
├── uncertain/
├── debug_images/
└── verification_report.json  # 통합 리포트
```

---

## 📋 사용 예시

### 예시 1: 4개 GPU 사용
```bash
# GPU 4개 자동 감지
./run_multigpu_simple.sh

# 출력 예시:
======================================================================
  YOLO Label Verifier - Multi-GPU Mode
  Using 4 GPUs
======================================================================
Total images: 1000
Image list split complete

Starting verification on 4 GPUs...

[GPU 0] Starting...
[GPU 1] Starting...
[GPU 2] Starting...
[GPU 3] Starting...

Waiting for all GPUs to complete...
[GPU 0] Complete! Processed 250 images
[GPU 1] Complete! Processed 250 images
[GPU 2] Complete! Processed 250 images
[GPU 3] Complete! Processed 250 images

Merging results...
✓ Results merged successfully!
  Total images: 1000
  Total boxes: 3542
  Correct: 2890
  Mislabeled: 512
  Uncertain: 140
```

### 예시 2: 2개 GPU만 사용
```bash
./run_multigpu_simple.sh 2
```

### 예시 3: Python wrapper 사용
```bash
# 기본 (GPU 자동 감지)
python multigpu_wrapper.py

# GPU 개수 지정
python multigpu_wrapper.py --num-gpus 4

# GPU별 output 보존
python multigpu_wrapper.py --keep-gpu-outputs

# 다른 config 사용
python multigpu_wrapper.py --config my_config.yaml
```

---

## ⚙️ 고급 옵션

### run_multigpu.sh (고급 버전)

```bash
# 환경 변수로 옵션 설정
export NUM_GPUS=4                # GPU 개수 지정
export KEEP_GPU_OUTPUTS=1        # GPU별 output 보존

./run_multigpu.sh
```

### multigpu_wrapper.py 옵션

```bash
python multigpu_wrapper.py --help

Options:
  --num-gpus N          GPU 개수 (기본: 자동 감지)
  --config PATH         Config 파일 경로
  --keep-gpu-outputs    GPU별 output 폴더 보존
```

---

## 🧪 테스트

### 기본 테스트
```bash
# 동의어 매칭 테스트
python test_aliases.py

# 멀티 GPU 동의어 테스트
python test_multigpu_aliases.py
```

### 소량 데이터 테스트
```bash
# 테스트용 작은 데이터셋으로 먼저 확인
# config.yaml에서 input 경로를 test_images로 변경
./run_multigpu_simple.sh
```

---

## 🔍 로그 확인

### 실시간 로그 모니터링
```bash
# GPU 0 로그
tail -f gpu0.log

# 모든 GPU 로그
tail -f gpu*.log
```

### 완료 후 로그 확인
```bash
# 에러 확인
grep -i error gpu*.log

# 통계 확인
grep -i "Total" gpu*.log
```

---

## 📈 성능 비교

### Single GPU
```
1000 images × 4초/image = 4000초 (약 67분)
```

### Multi GPU (4개)
```
250 images × 4초/image = 1000초 (약 17분)
속도 향상: 약 4배
```

**실제 속도 향상:**
- 2 GPU: 약 1.9배
- 4 GPU: 약 3.8배
- 8 GPU: 약 7.5배

*(오버헤드와 I/O로 인해 완벽한 선형 증가는 아님)*

---

## 🐛 문제 해결

### 1. "No GPUs detected"
```bash
# GPU 확인
nvidia-smi

# CUDA 경로 확인
echo $CUDA_HOME
which nvcc
```

### 2. "Permission denied"
```bash
# 실행 권한 부여
chmod +x run_multigpu_simple.sh
```

### 3. 일부 GPU만 실패
```bash
# 로그 확인
cat gpu2.log  # GPU 2 로그

# 특정 GPU만 사용
CUDA_VISIBLE_DEVICES=0,1,3 ./run_multigpu_simple.sh 3
```

### 4. "CUDA out of memory"
```bash
# config.yaml에서 배치 크기 조정
processing:
  batch_size: 1  # 기본값보다 줄이기
  
# 또는 더 적은 GPU 사용
./run_multigpu_simple.sh 2
```

### 5. 결과가 안 통합됨
```bash
# GPU별 output 확인
ls -la output_gpu*

# 수동 통합
python multigpu_wrapper.py  # 통합 스크립트 재실행
```

---

## 💡 팁

### 1. **최적 GPU 개수**
```
- 이미지 < 100개: Single GPU
- 이미지 100-500개: 2 GPU
- 이미지 500-2000개: 4 GPU
- 이미지 > 2000개: 8 GPU
```

### 2. **VRAM 모니터링**
```bash
# 실시간 VRAM 사용량 확인
watch -n 1 nvidia-smi
```

### 3. **병목 지점 확인**
```
- GPU 사용률 < 50%: CPU 병목 (이미지 로딩)
- GPU 사용률 > 90%: GPU 병목 (정상)
- VRAM 사용률 > 95%: 메모리 부족
```

### 4. **디스크 I/O 최적화**
```bash
# SSD 사용 권장
# NFS/네트워크 드라이브는 느림

# 이미지를 로컬 SSD로 먼저 복사
cp -r /network/images /local/ssd/images
```

---

## 📊 출력 구조

```
output/
├── correct/
│   ├── indoor/
│   │   ├── day/
│   │   │   ├── JPEGImages/
│   │   │   └── labels/
│   │   └── night/
│   └── outdoor/
├── mislabeled/
│   └── outdoor/
│       └── day/
│           ├── JPEGImages/
│           ├── labels/              # 수정된 라벨
│           └── labels_original/     # 원본 백업
├── uncertain/
├── debug_images/
│   ├── image001_verified.jpg
│   └── image002_verified.jpg
└── verification_report.json         # 통합 리포트
```

---

## 🔄 Single GPU로 다시 실행

```bash
# 기본 verifier.py 사용
python verifier.py
```

---

## 📞 문의

- 로그 파일: `gpu*.log`
- 통합 리포트: `output/verification_report.json`
- 개별 리포트: `output_gpu*/verification_report.json` (보존 시)

---

**추천: 처음 사용 시 소량 데이터로 테스트 후 전체 실행!** 🎯
