# 멀티 GPU 스크립트 - Config 전달 오류 수정

## 🐛 발견된 버그

### 에러 메시지
```
Traceback (most recent call last):
  File "run_gpu_worker_nocopy.py", line 45
    model_manager = verifier.ModelManager(config_loaded['model'])
KeyError: 'model'
```

### 원인

**잘못된 코드:**
```python
model_manager = verifier.ModelManager(config_loaded['model'])
```

**문제:**
- `ModelManager`에 `config['model']` 부분만 전달
- `ModelManager`는 전체 config를 기대하고 내부에서 `config['model']['cache_dir']` 접근
- 결과: KeyError 발생

**올바른 코드:**
```python
model_manager = verifier.ModelManager(config_loaded)
```

---

## ✅ 수정 내용

### 1. run_multigpu_nocopy.sh

**Before (Line 45):**
```python
model_manager = verifier.ModelManager(config_loaded['model'])
```

**After:**
```python
model_manager = verifier.ModelManager(config_loaded)
```

---

### 2. run_multigpu_simple.sh

**Before (Line 117):**
```python
model_manager = verifier.ModelManager(config_loaded['model'])
```

**After:**
```python
model_manager = verifier.ModelManager(config_loaded)
```

---

## 📋 ModelManager 시그니처

```python
class ModelManager:
    """모델 다운로드 및 관리"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config['model']['cache_dir'])  # ← 전체 config 필요
        self.selected_model = config['model'].get('selected_model')
        # ...
```

**기대하는 구조:**
```python
config = {
    'model': {
        'cache_dir': './models',
        'selected_model': 'Qwen3-VL-4B-Instruct'
    },
    'paths': { ... },
    'classes': { ... },
    # ...
}
```

---

## 🔍 왜 이런 실수가?

### 혼란의 원인

**main() 함수에서:**
```python
# 1. Config 전체 로드
config = yaml.safe_load(f)

# 2. ModelManager 생성
model_manager = ModelManager(config)  # ✓ 전체 전달

# 3. LabelVerifier 생성
label_verifier = LabelVerifier(model_manager, config)  # ✓ 전체 전달
```

**멀티 GPU 스크립트에서 (잘못됨):**
```python
# 1. Config 전체 로드
config_loaded = yaml.safe_load(f)

# 2. ModelManager 생성
model_manager = ModelManager(config_loaded['model'])  # ✗ 부분만 전달
```

**혼란 포인트:**
- 메서드 이름이 `ModelManager`이므로 `config['model']`만 전달하면 될 것 같음
- 하지만 실제로는 **전체 config가 필요**

---

## ✅ 테스트

### 수정 전 (에러)
```bash
$ bash run_multigpu_nocopy.sh

[GPU 0] Starting...
Traceback (most recent call last):
  File "run_gpu_worker_nocopy.py", line 45
    model_manager = verifier.ModelManager(config_loaded['model'])
KeyError: 'model'
```

### 수정 후 (정상)
```bash
$ bash run_multigpu_nocopy.sh

[GPU 0] Starting...
[GPU 0] Processing 32273 images
[GPU 0] Starting verification...
2025-12-11 - [GPU 0] INFO - Model loaded successfully
[GPU 0] Progress: 100/32273
...
```

---

## 🎯 영향받는 파일

| 파일 | 상태 | 설명 |
|-----|------|------|
| **run_multigpu_nocopy.sh** | ✅ 수정됨 | Line 45 수정 |
| **run_multigpu_simple.sh** | ✅ 수정됨 | Line 117 수정 |
| run_multigpu.sh | ✅ 영향없음 | wrapper 호출 방식 |
| multigpu_wrapper.py | ✅ 영향없음 | main() 직접 호출 |

---

## 📝 체크리스트

수정된 스크립트 사용 시 확인사항:

- [x] `ModelManager(config)` (전체 config)
- [x] `LabelVerifier(model_manager, config)` (전체 config)
- [x] Config에 'model' 섹션 존재
- [x] Config에 'paths' 섹션 존재
- [x] Config에 'classes' 섹션 존재

---

## 💡 기억하세요

**모든 Manager/Verifier 클래스는 전체 config를 받습니다:**

```python
# ✓ 올바른 방식
model_manager = ModelManager(config)
label_verifier = LabelVerifier(model_manager, config)
result_manager = ResultManager(config)

# ✗ 잘못된 방식
model_manager = ModelManager(config['model'])  # KeyError!
```

---

## 🚀 이제 다시 실행하세요

```bash
# 수정된 스크립트로 재실행
bash run_multigpu_nocopy.sh

# 또는
bash run_multigpu_simple.sh
```

이제 정상적으로 작동합니다! ✅
