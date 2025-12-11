# 오프라인 환경 설정 가이드

오프라인 환경에서 이미 다운로드된 모델을 사용하는 방법입니다.

---

## 🔍 문제 증상

```
Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)
Max retries exceeded with url: /Qwen/Qwen3-VL-8B-Instruct/resolve/main/config.json
```

**원인:**
- 로컬에 모델이 있음
- 하지만 Hugging Face가 온라인으로 최신 버전 확인 시도
- 오프라인 환경이라 실패

---

## ✅ 해결 방법

### **방법 1: verifier.py 수정** ⭐ (영구적)

**수정 위치:** `load_model()` 메서드

**Before:**
```python
self.model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    cache_dir=str(self.cache_dir),
    trust_remote_code=True
)

self.processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=str(self.cache_dir)
)
```

**After:**
```python
self.model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    cache_dir=str(self.cache_dir),
    trust_remote_code=True,
    local_files_only=True  # ← 추가!
)

self.processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=str(self.cache_dir),
    local_files_only=True  # ← 추가!
)
```

---

### **방법 2: 환경 변수 설정** (임시)

```bash
# 실행 전 설정
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 그 다음 실행
bash run_multigpu_v2.sh
```

**멀티 GPU 스크립트 자동 설정:**

`run_multigpu_v2.sh`에 이미 포함됨:
```python
# 오프라인 모드 설정
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

---

### **방법 3: 수동 패치**

```bash
# verifier.py 백업
cp verifier.py verifier.py.backup

# 자동 패치
sed -i '/from_pretrained(/,/^[[:space:]]*)/s/)$/,\n                local_files_only=True\n            )/' verifier.py
```

---

## 🔧 적용 방법

### **Step 1: verifier.py 업데이트**

```bash
# 다운로드한 최신 verifier.py 사용
# 또는 수동 수정
```

**확인:**
```bash
grep -A2 "from_pretrained" verifier.py | grep "local_files_only"
```

**출력이 이렇게 나와야 함:**
```
                local_files_only=True  # 오프라인 모드 강제
```

---

### **Step 2: 테스트**

```bash
# 환경 변수 설정 (추가 보험)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 실행
bash run_multigpu_v2.sh
```

---

## 📋 모델 캐시 구조 확인

### **Hugging Face 캐시 형식**

```
models/
├── models--Qwen--Qwen3-VL-4B-Instruct/
│   ├── blobs/
│   │   └── (모델 파일들)
│   ├── refs/
│   │   └── main
│   └── snapshots/
│       └── (해시)/
│           ├── config.json
│           ├── model.safetensors
│           └── ...
└── models--Qwen--Qwen3-VL-8B-Instruct/
    └── ...
```

**확인 명령:**
```bash
ls -la models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/config.json
```

**있으면 ✓** - 오프라인 모드 사용 가능

---

## 🚨 문제 해결

### Q: 여전히 온라인 접속 시도

**확인:**
```bash
grep "local_files_only" verifier.py
```

**없으면:**
```bash
# verifier.py가 최신 버전이 아님
# 다시 다운로드하거나 수동 수정
```

---

### Q: "config.json not found" 에러

**원인:** 모델이 완전히 다운로드되지 않음

**해결:**
```bash
# 1. 모델 구조 확인
ls -la models/models--Qwen--Qwen3-VL-8B-Instruct/

# 2. snapshots 폴더 확인
ls -la models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/

# 3. config.json 찾기
find models/ -name "config.json"

# 4. 없으면 온라인 환경에서 다시 다운로드 필요
```

---

### Q: "model.safetensors not found" 에러

**확인:**
```bash
# 모델 파일 확인
find models/models--Qwen--Qwen3-VL-8B-Instruct/ -name "*.safetensors"
```

**있어야 하는 파일들:**
```
model.safetensors
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

---

## 💡 모델 다운로드 (온라인 환경)

오프라인으로 가기 전에 모델을 완전히 다운로드하세요:

```bash
# 온라인 환경에서
python3 << EOF
from transformers import AutoModelForVision2Seq, AutoProcessor

model_id = "Qwen/Qwen3-VL-8B-Instruct"
cache_dir = "./models"

print("Downloading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype="auto"
)

print("Downloading processor...")
processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir
)

print("✓ Download complete!")
EOF
```

---

## 🎯 전체 체크리스트

오프라인 환경 준비:

- [ ] 모델 완전히 다운로드됨
- [ ] `config.json` 존재 확인
- [ ] `model.safetensors` 존재 확인
- [ ] `verifier.py`에 `local_files_only=True` 추가됨
- [ ] 환경 변수 `HF_HUB_OFFLINE=1` 설정
- [ ] 테스트 실행 성공

---

## 📝 실행 예시

### **온라인 접속 시도 (Before)**
```
2025-12-11 09:10:09,780 - [GPU 0] WARNING - Failed to resolve 'huggingface.co'
Retrying in 1s [Retry 1/5]...
Retrying in 2s [Retry 2/5]...
(계속 재시도...)
```

### **오프라인 모드 (After)**
```
2025-12-11 09:15:30,123 - [GPU 0] INFO - Loading model: Qwen/Qwen3-VL-8B-Instruct
2025-12-11 09:15:30,124 - [GPU 0] INFO - Cache directory: ./models
2025-12-11 09:15:35,456 - [GPU 0] INFO - ✓ Model loaded successfully
2025-12-11 09:15:35,457 - [GPU 0] INFO - GPU Memory - Allocated: 15.23GB
[GPU 0] Progress: 312/31227 (1.0%) | Speed: 4.2 img/s | ETA: 120.5 min
```

---

## 🔗 참고

**Hugging Face 오프라인 문서:**
- https://huggingface.co/docs/transformers/installation#offline-mode

**환경 변수:**
- `HF_HUB_OFFLINE=1` - Hugging Face Hub 오프라인 모드
- `TRANSFORMERS_OFFLINE=1` - Transformers 라이브러리 오프라인 모드

---

**이제 완전한 오프라인 환경에서 작동합니다!** 🎉
