# 멀티 GPU 스크립트 v2 - 개선사항

## 🔍 기존 문제점

### 1. 진행 상황이 화면에 안 보임
```bash
# 기존 코드
python3 ... > gpu${i}.log 2>&1 &
#             ^^^^^^^^^^^^^^^^^^^
#             파일로만 리다이렉트
```

**증상:**
```
[GPU 0] Starting...
[GPU 1] Starting...
...
(아무것도 안 보임) ← 1시간 대기...
```

**원인:**
- 백그라운드 실행 (`&`)
- stdout/stderr를 파일로만 리다이렉트
- 로그 파일을 직접 열어봐야 진행 상황 확인 가능

---

### 2. GPU별 폴더가 남아있음
```
output_gpu0/
output_gpu1/
output_gpu2/
...
output/  ← 최종 결과
```

**원인:**
```python
config['paths']['output_base'] = config['paths']['output_base'] + f'_gpu{gpu_id}'
```

**흐름:**
1. 각 GPU가 `output_gpu{N}`에 씀
2. 나중에 모두 `output`으로 merge
3. cleanup에서 `output_gpu*` 삭제 (하지만 실행 안 되거나 중간 상태)

---

## ✅ v2 개선사항

### 1. 진행 상황 실시간 표시

**Before:**
```bash
python3 ... > gpu${i}.log 2>&1 &
```

**After:**
```bash
python3 ... 2>&1 | tee gpu${i}.log &
#                  ^^^
#                  화면과 파일 동시 출력
```

**결과:**
```
[GPU 0] Progress: 100/32273 (0.3%) | Speed: 4.2 img/s | ETA: 127.5 min
[GPU 1] Progress: 150/32273 (0.5%) | Speed: 4.5 img/s | ETA: 119.2 min
[GPU 2] Progress: 80/32273 (0.2%) | Speed: 3.8 img/s | ETA: 141.3 min
...
```

---

### 2. GPU별 폴더 제거

**Before:**
```python
config['paths']['output_base'] = config['paths']['output_base'] + f'_gpu{gpu_id}'
```

**After:**
```python
# GPU별 폴더 없음
# 각 GPU가 직접 output에 씀
```

**폴더 구조:**
```
output/
├── correct/
│   ├── JPEGImages/
│   └── labels/
├── mislabeled/
│   ├── JPEGImages/
│   └── labels/
└── uncertain/

# GPU별 폴더 없음!
```

**충돌 방지:**
```python
target_folder.mkdir(parents=True, exist_ok=True)
#                                 ^^^^^^^^^^^^^
#                                 race condition 방지
```

---

### 3. 더 자주 업데이트

**Before:**
```python
if (idx + 1) % 10 == 0:  # 10개마다
    print(f"Progress: {idx + 1}/{total}")
```

**After:**
```python
progress_interval = max(1, len(image_paths) // 100)  # 1%마다

if (idx + 1) % progress_interval == 0:
    print(progress_msg, flush=True)
```

**예시:**
- 32,273 images → 323개마다 업데이트
- 100 images → 1개마다 업데이트

---

### 4. ETA 표시

```python
elapsed = (datetime.now() - start_time).total_seconds()
speed = processed_count / elapsed
remaining = (total - processed) / speed

progress_msg = (
    f"[GPU {gpu_id}] Progress: {idx + 1}/{total} "
    f"({100*(idx+1)/total:.1f}%) | "
    f"Speed: {speed:.1f} img/s | "
    f"ETA: {remaining/60:.1f} min"
)
```

---

### 5. 임시 리포트 방식

**Before:**
- 각 GPU가 `output_gpu{N}/verification_report.json` 생성
- Merge 시 모두 읽어서 통합
- cleanup에서 `output_gpu*` 폴더 삭제

**After:**
- 각 GPU가 `temp_report_gpu{N}.json` 생성 (루트 폴더)
- Merge 시 읽어서 통합
- 임시 파일만 삭제

**장점:**
- GPU별 폴더 불필요
- 더 빠름 (파일 복사 없음)
- 깔끔함

---

## 📊 실행 예시

### v1 (기존)
```bash
$ bash run_multigpu_nocopy.sh

======================================================================
  YOLO Label Verifier - Multi-GPU Mode (No Copy)
  Using 8 GPUs
======================================================================
Total images: 258180
Image list split complete

Starting verification on 8 GPUs...

[GPU 0] Starting...
[GPU 1] Starting...
...
Waiting for all GPUs to complete...

(1시간 대기... 아무것도 안 보임)

======================================================================
  All GPUs completed!
======================================================================
```

### v2 (개선)
```bash
$ bash run_multigpu_v2.sh

======================================================================
  YOLO Label Verifier - Multi-GPU Mode
  Using 8 GPUs
======================================================================
Total images: 258180
Image list split complete

Starting verification on 8 GPUs...

Progress will be displayed below:
======================================================================

[GPU 0] Processing 32273 images
[GPU 1] Processing 32273 images
...
[GPU 0] Progress: 323/32273 (1.0%) | Speed: 4.2 img/s | ETA: 127.5 min
[GPU 2] Progress: 323/32272 (1.0%) | Speed: 4.1 img/s | ETA: 130.1 min
[GPU 1] Progress: 323/32273 (1.0%) | Speed: 4.5 img/s | ETA: 119.2 min
...
[GPU 0] Progress: 646/32273 (2.0%) | Speed: 4.3 img/s | ETA: 123.4 min
[GPU 5] Progress: 646/32272 (2.0%) | Speed: 4.4 img/s | ETA: 120.8 min
...
(실시간 업데이트 계속...)
...
[GPU 0] ✓ Complete! Processed 32273 images
[GPU 1] ✓ Complete! Processed 32273 images
...

======================================================================
  All GPUs completed!
======================================================================

Merging results...
Merged 258180 images from 8 GPUs

✓ Results merged successfully!
  Output: /workspace/datasets/db2/101.etc/solbrain/data/output
  Total images: 258180
  Total boxes: 1234567
  Correct: 980000
  Mislabeled: 180000
  Uncertain: 74567

Cleaning up temporary files...

======================================================================
✓ Multi-GPU verification complete!
======================================================================
```

---

## 🔄 차이점 요약

| 항목 | v1 (기존) | v2 (개선) |
|-----|----------|----------|
| **화면 출력** | Starting만 | 실시간 Progress |
| **GPU 폴더** | output_gpu{N} 생성 | 생성 안 함 |
| **업데이트 주기** | 10개마다 | 1%마다 |
| **ETA** | 없음 | 표시됨 |
| **속도 표시** | 없음 | img/s |
| **임시 파일** | 폴더 전체 | JSON만 |

---

## 🚀 사용 방법

### 기존 스크립트 확인
```bash
ls -la output_gpu*/  # GPU별 폴더가 있는지 확인

# 있다면 정리
rm -rf output_gpu*/
```

### v2 실행
```bash
bash run_multigpu_v2.sh

# 또는 GPU 개수 지정
bash run_multigpu_v2.sh 4
```

### 별도 터미널에서 모니터링 (선택사항)
```bash
# 로그 파일도 동시에 생성되므로
tail -f gpu0.log

# 또는 모니터링 스크립트
bash monitor_progress.sh
```

---

## 🎯 장점

### 1. 답답함 해소
- **Before**: 1시간 동안 아무것도 안 보임
- **After**: 매 1%마다 업데이트, ETA 표시

### 2. 폴더 깔끔
- **Before**: output_gpu0, output_gpu1, ... 생성됨
- **After**: output 폴더만 존재

### 3. 디버깅 쉬움
- 화면에서 바로 에러 확인 가능
- 로그 파일도 계속 생성됨

### 4. 빠른 피드백
- GPU별 속도 비교 가능
- 느린 GPU 즉시 확인

---

## ⚠️ 주의사항

### 1. 화면 출력 많음
- 8 GPUs × 100 updates = 800줄
- 터미널 스크롤 버퍼 주의

### 2. 로그 파일 동시 생성
- `gpu0.log`, `gpu1.log`, ... 계속 생성
- 화면 출력과 동일한 내용

### 3. 중단 시
```bash
# Ctrl+C로 중단하면
# 임시 파일 수동 정리 필요
rm -f temp_report_gpu*.json
rm -f split_*
```

---

## 🔧 문제 해결

### Q: 여전히 화면에 안 보임
```bash
# Python buffering 문제일 수 있음
# 스크립트에 flush=True 추가됨
print(msg, flush=True)
```

### Q: GPU별 폴더가 여전히 생김
```bash
# v2 스크립트 사용 확인
bash run_multigpu_v2.sh  # ← v2!

# 또는
ls -la run_multigpu_v2.sh
```

### Q: 충돌 발생
```bash
# 매우 드물지만 발생 가능
# 로그 확인
grep -i "error\|exception" gpu*.log
```

---

## 📝 기술적 세부사항

### tee 명령어
```bash
command 2>&1 | tee output.log &
#              ^^^
#              stdout/stderr를 화면과 파일에 동시 출력
```

### exist_ok=True
```python
Path("folder").mkdir(parents=True, exist_ok=True)
#                                   ^^^^^^^^^^^^^
#                                   이미 있어도 에러 안 남
```

### flush=True
```python
print(msg, flush=True)
#         ^^^^^^^^^^^
#         즉시 출력 (버퍼링 없음)
```

---

## ✅ 체크리스트

v2로 마이그레이션:

- [ ] 기존 GPU별 폴더 정리: `rm -rf output_gpu*/`
- [ ] v2 스크립트 실행: `bash run_multigpu_v2.sh`
- [ ] 화면에서 Progress 확인
- [ ] 완료 후 output 폴더만 남아있는지 확인
- [ ] GPU별 폴더 없는지 확인: `ls output_gpu*` → 없어야 함

---

**이제 진행 상황이 실시간으로 보이고, 폴더 구조도 깔끔합니다!** 🎉
