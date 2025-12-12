# VLM 프롬프트 자동 최적화 시스템

테스트 DB를 사용하여 각 클래스별로 최적의 VLM 프롬프트를 자동으로 찾는 시스템입니다.

## 📌 핵심 개념

- **VLM 메타인지 의존 없음**: VLM에게 "어떤 프롬프트가 좋니?"라고 묻지 않음
- **실제 측정 기반**: 다양한 프롬프트 후보를 GT 샘플로 직접 테스트
- **정량적 선택**: 정확도가 가장 높은 프롬프트를 선택

## 🏗️ 시스템 구조

```
prompt_discovery/
├── gt_loader.py           # Ground Truth 로더
├── prompt_generator.py    # 프롬프트 후보 생성기
├── prompt_tester.py       # 프롬프트 테스터
├── result_analyzer.py     # 결과 분석기
│
├── scripts/
│   ├── prepare_test_db.py    # GT 준비 스크립트
│   └── optimize_prompts.py   # 최적화 실행 스크립트
│
├── test_db/
│   ├── JPEGImages/        # GT 이미지 (YOLO 표준 구조)
│   └── labels/            # GT 라벨 (YOLO format)
│
├── results/               # 테스트 결과
└── templates/             # 최적화된 프롬프트 템플릿
    └── current.yaml       # 현재 사용 중인 템플릿
```

## 🚀 사용 방법

### 1단계: 테스트 DB 준비

프로덕션 데이터에서 클래스별로 샘플을 추출합니다.

```bash
# config.yaml의 경로 사용
python3 prompt_discovery/scripts/prepare_test_db.py \
  --samples 30 \
  --classes 0 3 6  # person, helmet, car만

# 또는 직접 경로 지정
python3 prompt_discovery/scripts/prepare_test_db.py \
  --source-images /path/to/images \
  --source-labels /path/to/labels \
  --output prompt_discovery/test_db \
  --samples 30
```

**결과:**
- `prompt_discovery/test_db/images/` - GT 이미지
- `prompt_discovery/test_db/labels/` - GT 라벨
- `prompt_discovery/test_db/summary.yaml` - 요약 정보

### 2단계: 프롬프트 최적화

테스트 DB를 사용하여 최적의 프롬프트를 찾습니다.

```bash
# 단일 클래스
python3 prompt_discovery/scripts/optimize_prompts.py --class helmet

# 여러 클래스
python3 prompt_discovery/scripts/optimize_prompts.py --classes helmet person car

# 모든 클래스
python3 prompt_discovery/scripts/optimize_prompts.py --all

# 진행률 표시 없이 (로그 파일 저장 시)
python3 prompt_discovery/scripts/optimize_prompts.py --all --no-progress > optimize.log 2>&1
```

**실행 과정:**
1. GT 샘플 로드 (클래스별)
2. 프롬프트 후보 생성 (20~30개)
3. 각 프롬프트를 GT로 테스트
4. 정확도 측정 및 순위 매기기
5. 최적 프롬프트 저장

**결과:**
- `prompt_discovery/results/{class}_results.json` - 클래스별 결과
- `prompt_discovery/templates/optimized_YYYYMMDD_HHMMSS.yaml` - 최적화된 템플릿
- `prompt_discovery/templates/current.yaml` - 심볼릭 링크 (최신 템플릿)

### 3단계: 프로덕션 적용

최적화된 프롬프트를 프로덕션에 적용합니다.

**config.yaml 수정:**
```yaml
prompt_optimization:
  production:
    use_optimized_prompts: true  # 활성화
```

**verifier.py 실행:**
```bash
python3 verifier.py  # 최적화된 프롬프트 자동 사용
```

## 📊 프롬프트 후보 종류

프롬프트 생성기는 다음과 같은 다양한 형식의 프롬프트를 생성합니다:

### 1. 기본 질문 형식
```
Is this a helmet? Answer Yes or No.
Is there a helmet in this image? Answer Yes or No.
Does this image contain a helmet? Answer Yes or No.
```

### 2. 디테일 강조 형식
```
Looking at this image carefully, is this a helmet? Answer Yes or No.
Based on the visual features, is this a helmet? Answer Yes or No.
```

### 3. 컨텍스트 포함 형식
```
Is this helmet (safety equipment) present? Answer Yes or No.
Looking at this as safety equipment, is it a helmet? Answer Yes or No.
```

### 4. 부정 질문 형식
```
Is this NOT a helmet? Answer Yes or No.
```

### 5. 설명 요청 형식
```
Identify if this is a helmet. Reply only Yes or No.
Determine whether this is a helmet. Answer Yes or No.
```

## 📈 결과 예시

```json
{
  "class_name": "helmet",
  "best_prompt": {
    "prompt": "Based on the visual features, is this a helmet? Answer Yes or No.",
    "accuracy": 0.933,
    "avg_confidence": 0.876,
    "correct_count": 28,
    "total_count": 30
  },
  "statistics": {
    "best_accuracy": 0.933,
    "worst_accuracy": 0.600,
    "avg_accuracy": 0.782
  }
}
```

## 🎯 추천 워크플로우

### 초기 설정 (1회)
```bash
# 1. 테스트 DB 준비 (클래스당 30개 샘플)
python3 prompt_discovery/scripts/prepare_test_db.py --samples 30 --all

# 2. 모든 클래스 최적화
python3 prompt_discovery/scripts/optimize_prompts.py --all

# 3. config.yaml 수정
vim config.yaml  # use_optimized_prompts: true
```

### 정기 업데이트 (월 1회)
```bash
# 1. 새로운 데이터로 테스트 DB 갱신
python3 prompt_discovery/scripts/prepare_test_db.py --samples 30 --seed 123

# 2. 재최적화
python3 prompt_discovery/scripts/optimize_prompts.py --all

# 3. 성능 비교
diff prompt_discovery/templates/optimized_OLD.yaml \
     prompt_discovery/templates/optimized_NEW.yaml
```

### 새 클래스 추가 시
```bash
# 1. 새 클래스만 최적화
python3 prompt_discovery/scripts/optimize_prompts.py --class new_class_name

# 2. 템플릿 자동 업데이트 (기존 클래스 유지)
```

## 🔧 커스터마이징

### 사용자 정의 프롬프트 추가

`prompt_generator.py`에서 커스텀 프롬프트 추가:

```python
generator = PromptGenerator(config)

custom_prompts = [
    "Is this a safety helmet worn on construction sites? Yes/No",
    "Hard hat detected? Yes/No",
    "PPE: helmet present? Yes/No"
]

all_prompts = generator.add_custom_candidates('helmet', custom_prompts)
```

### 테스트 샘플 수 조정

더 많은 샘플 = 더 정확한 측정 (하지만 느림)

```bash
# 클래스당 50개 샘플
python3 prompt_discovery/scripts/prepare_test_db.py --samples 50
```

### 프롬프트 후보 수 제한

`config.yaml`:
```yaml
prompt_optimization:
  testing:
    max_candidates: 20  # 기본값: 30
```

## 📋 요구사항

- Python 3.8+
- PyTorch
- Transformers
- Qwen3-VL 모델
- 기존 verifier.py 시스템

## ⚠️ 주의사항

1. **GT 품질**: 테스트 DB의 라벨이 정확해야 함
2. **계산 비용**: 프롬프트 30개 × 샘플 30개 = 900회 VLM 호출
3. **클래스 균형**: 모든 클래스에 충분한 샘플 필요 (최소 10개)
4. **Confidence Threshold**: 기존 시스템과 동일한 threshold 사용 (0.6)

## 🐛 문제 해결

### "No GT samples found"
```bash
# GT 디렉토리 확인
ls prompt_discovery/test_db/images/
ls prompt_discovery/test_db/labels/

# 클래스 분포 확인
python3 -c "
import sys; sys.path.insert(0, 'prompt_discovery')
from gt_loader import GroundTruthLoader
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
loader = GroundTruthLoader(config)
print(loader.get_class_distribution())
"
```

### "Model loading failed"
```bash
# 모델 캐시 확인
ls models/

# 오프라인 모드 비활성화
export HF_HUB_OFFLINE=0
```

### 성능이 기대보다 낮음
- GT 라벨 정확도 재확인
- 더 많은 샘플로 테스트
- 클래스별 컨텍스트 힌트 추가

## 📚 참고

- [verifier.py](../verifier.py) - 메인 검증 시스템
- [config.yaml](../config.yaml) - 전체 설정
- [MULTIGPU_GUIDE.md](../MULTIGPU_GUIDE.md) - Multi-GPU 실행 가이드
