# VLM 프롬프트 자동 최적화 가이드

## 🎯 개요

이 시스템은 테스트 DB를 사용하여 각 클래스별로 최적의 VLM 프롬프트를 자동으로 찾아서 적용합니다.

**핵심 장점:**
- VLM 메타인지에 의존하지 않음 (실제 측정 기반)
- 정량적 선택 (정확도로 평가)
- 클래스별 맞춤 프롬프트

---

## 🚀 빠른 시작 (5분)

### 1. 테스트 DB 준비
```bash
# 기존 데이터에서 샘플 추출 (클래스당 20개)
python3 prompt_discovery/scripts/prepare_test_db.py --samples 20
```

### 2. 프롬프트 최적화 (helmet 클래스)
```bash
# 단일 클래스 테스트
python3 prompt_discovery/scripts/optimize_prompts.py --class helmet
```

### 3. 결과 확인
```bash
# 결과 파일 확인
cat prompt_discovery/results/helmet_results.json

# 최적 프롬프트 확인
cat prompt_discovery/templates/current.yaml
```

### 4. 프로덕션 적용
```yaml
# config.yaml 수정
prompt_optimization:
  production:
    use_optimized_prompts: true
```

```bash
# 검증 실행 (최적화된 프롬프트 자동 사용)
python3 verifier.py
```

---

## 📚 상세 가이드

### Step 1: 테스트 DB 준비

#### 옵션 A: config.yaml 경로 사용
```bash
python3 prompt_discovery/scripts/prepare_test_db.py \
  --samples 30 \
  --classes 0 3 6  # person, helmet, car만
```

#### 옵션 B: 직접 경로 지정
```bash
python3 prompt_discovery/scripts/prepare_test_db.py \
  --source-images /path/to/images \
  --source-labels /path/to/labels \
  --output prompt_discovery/test_db \
  --samples 30
```

**결과:**
```
prompt_discovery/test_db/
├── JPEGImages/    # GT 이미지 (클래스당 30개, YOLO 표준)
├── labels/        # GT 라벨 (YOLO format)
└── summary.yaml   # 샘플 정보
```

**GT 품질 확인:**
```bash
# 클래스 분포 확인
python3 -c "
import sys, yaml
sys.path.insert(0, 'prompt_discovery')
from gt_loader import GroundTruthLoader
with open('config.yaml') as f: config = yaml.safe_load(f)
loader = GroundTruthLoader(config)
for cls, cnt in loader.get_class_distribution().items():
    print(f'{cls}: {cnt} samples')
"
```

### Step 2: 프롬프트 최적화

#### 단일 클래스
```bash
python3 prompt_discovery/scripts/optimize_prompts.py --class helmet
```

**실행 과정:**
1. GT 로드 (helmet 샘플만)
2. 프롬프트 후보 생성 (20~30개)
3. 각 프롬프트를 GT로 테스트
4. 정확도 측정 및 순위 매기기
5. 결과 저장

**예상 시간:**
- 샘플 30개 × 프롬프트 25개 = 750회 VLM 호출
- GPU 1개: 약 15~30분 (모델 크기에 따라)

#### 여러 클래스
```bash
python3 prompt_discovery/scripts/optimize_prompts.py --classes helmet person car
```

#### 모든 클래스
```bash
python3 prompt_discovery/scripts/optimize_prompts.py --all
```

#### 로그 저장
```bash
python3 prompt_discovery/scripts/optimize_prompts.py --all --no-progress > optimize.log 2>&1
```

### Step 3: 결과 분석

#### 결과 파일 구조
```
prompt_discovery/
├── results/
│   ├── helmet_results.json     # 요약 결과
│   └── helmet_detailed.json    # 상세 결과
└── templates/
    ├── optimized_20251213_143022.yaml  # 최적화된 템플릿
    └── current.yaml                     # 심볼릭 링크 (최신)
```

#### 결과 해석
```json
{
  "class_name": "helmet",
  "best_prompt": {
    "prompt": "Based on the visual features, is this a helmet? Answer Yes or No.",
    "accuracy": 0.933,              // 93.3% 정확도
    "avg_confidence": 0.876,        // 평균 confidence 87.6%
    "correct_count": 28,
    "total_count": 30
  },
  "top_prompts": [
    {
      "rank": 1,
      "prompt": "...",
      "accuracy": 0.933
    },
    {
      "rank": 2,
      "prompt": "...",
      "accuracy": 0.900
    }
  ]
}
```

**성능 지표:**
- **accuracy**: 정확도 (0.0~1.0)
  - 0.9 이상: 우수
  - 0.8~0.9: 양호
  - 0.8 미만: 개선 필요

- **avg_confidence**: 평균 confidence (0.0~1.0)
  - 높을수록 확신도가 높음
  - 낮으면 모델이 불확실

#### 프롬프트 비교
```bash
# 상위 3개 프롬프트 비교
python3 -c "
import json
with open('prompt_discovery/results/helmet_results.json') as f:
    data = json.load(f)
    for p in data['top_prompts'][:3]:
        print(f'[{p[\"rank\"]}] Acc: {p[\"accuracy\"]:.3f} - {p[\"prompt\"][:60]}')
"
```

### Step 4: 프로덕션 적용

#### config.yaml 수정
```yaml
prompt_optimization:
  enabled: true  # 시스템 활성화

  production:
    use_optimized_prompts: true  # 최적화된 프롬프트 사용
    fallback_prompt: "Is this object a {class_name}? Answer Yes or No."
```

#### verifier.py 실행
```bash
# 기존과 동일하게 실행
python3 verifier.py

# Multi-GPU도 동일
./run_multigpu_v2.sh
```

**로그에서 확인:**
```
Loaded optimized prompt for 'helmet' (acc: 0.933)
Loaded optimized prompt for 'person' (acc: 0.887)
✓ Loaded 2 optimized prompts
```

#### 성능 모니터링
```bash
# 검증 후 리포트 확인
cat F:/solbrain/data_1210/output/verification_report.json | jq '.statistics'
```

---

## 🔧 고급 사용법

### 커스텀 프롬프트 추가

`prompt_discovery/scripts/test_custom.py`:
```python
from prompt_generator import PromptGenerator
from prompt_tester import PromptTester
import yaml

# Config 로드
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 커스텀 프롬프트
custom_prompts = [
    "Is this a safety helmet worn on construction sites? Yes/No",
    "Hard hat detected? Yes/No",
    "PPE: helmet present? Yes/No"
]

# 프롬프트 생성기
generator = PromptGenerator(config)
all_prompts = generator.add_custom_candidates('helmet', custom_prompts)

# 테스트 (verifier, loader 필요)
# ... (optimize_prompts.py와 동일)
```

### 클래스별 최소 샘플 수 설정

```yaml
# config.yaml
prompt_optimization:
  ground_truth:
    min_samples_per_class: 20  # 20개 미만이면 경고
```

### 프롬프트 후보 수 제한

```yaml
# config.yaml
prompt_optimization:
  testing:
    max_candidates: 20  # 기본값: 30 (빠르게 테스트)
    save_top_n: 5       # 상위 5개만 저장
```

### 특정 템플릿 사용

```bash
# 이전 템플릿 복원
cd prompt_discovery/templates
rm current.yaml
ln -s optimized_20251201_120000.yaml current.yaml
```

---

## 📊 성능 비교

### A/B 테스트

#### 1. 기존 프롬프트 성능 측정
```bash
# use_optimized_prompts: false로 실행
python3 verifier.py
# 결과 저장: baseline_report.json
```

#### 2. 최적화된 프롬프트 성능 측정
```bash
# use_optimized_prompts: true로 실행
python3 verifier.py
# 결과 저장: optimized_report.json
```

#### 3. 비교
```python
import json

with open('baseline_report.json') as f:
    baseline = json.load(f)

with open('optimized_report.json') as f:
    optimized = json.load(f)

for cls in baseline['statistics']['per_class'].keys():
    base_acc = baseline['statistics']['per_class'][cls]['correct'] / max(1, sum(baseline['statistics']['per_class'][cls].values()))
    opt_acc = optimized['statistics']['per_class'][cls]['correct'] / max(1, sum(optimized['statistics']['per_class'][cls].values()))

    improvement = opt_acc - base_acc
    print(f"{cls}: {base_acc:.3f} → {opt_acc:.3f} ({improvement:+.3f})")
```

---

## ⚠️ 주의사항

### 1. GT 품질
- 테스트 DB의 라벨이 **100% 정확**해야 함
- 잘못된 라벨 → 잘못된 프롬프트 선택

**검증 방법:**
```bash
# GT를 기존 시스템으로 검증
python3 verifier.py  # test_db를 input으로 설정
# mislabeled가 많으면 GT 재확인 필요
```

### 2. 샘플 수
- 클래스당 **최소 20개** 권장
- 너무 적으면 통계적으로 불안정
- 너무 많으면 시간 오래 걸림

### 3. 계산 비용
- 프롬프트 30개 × 샘플 30개 = **900회 VLM 호출**
- 클래스 10개 = **9,000회 호출**
- GPU 1개: 약 3~5시간

### 4. 오버피팅
- GT에만 최적화됨
- 실제 데이터에서 성능 검증 필요

---

## 🐛 문제 해결

### "No GT samples found"
```bash
# GT 확인
ls prompt_discovery/test_db/JPEGImages/  # Linux
dir prompt_discovery\test_db\JPEGImages  # Windows
ls prompt_discovery/test_db/labels/

# 클래스 분포 확인
python3 prompt_discovery/gt_loader.py
```

### "Model loading failed"
```bash
# 모델 캐시 확인
ls models/

# 오프라인 모드 해제
export HF_HUB_OFFLINE=0
```

### 최적화 후 성능이 더 나빠짐
1. GT 품질 재확인
2. 샘플 수 늘리기 (30→50)
3. 다른 클래스도 함께 최적화

### "Template not found"
```bash
# 템플릿 생성 확인
ls prompt_discovery/templates/

# 수동 생성
python3 prompt_discovery/scripts/optimize_prompts.py --class helmet
```

---

## 📈 모범 사례

### 초기 설정 (1회)
```bash
# 1. 대표 클래스로 테스트 (helmet, person)
python3 prompt_discovery/scripts/prepare_test_db.py --samples 30 --classes 0 3

# 2. 프롬프트 최적화
python3 prompt_discovery/scripts/optimize_prompts.py --classes helmet person

# 3. 결과 확인
cat prompt_discovery/results/helmet_results.json

# 4. A/B 테스트
# - baseline vs optimized 비교
# - 실제 데이터로 검증

# 5. 전체 클래스 확대
python3 prompt_discovery/scripts/optimize_prompts.py --all
```

### 정기 업데이트 (월 1회)
```bash
# 1. 새로운 데이터로 GT 갱신
python3 prompt_discovery/scripts/prepare_test_db.py --samples 30 --seed 999

# 2. 재최적화
python3 prompt_discovery/scripts/optimize_prompts.py --all

# 3. 성능 비교
diff prompt_discovery/templates/optimized_OLD.yaml \
     prompt_discovery/templates/current.yaml
```

### 새 클래스 추가 시
```bash
# 1. 새 클래스만 최적화
python3 prompt_discovery/scripts/optimize_prompts.py --class new_class

# 2. 템플릿 자동 병합 (기존 유지)
```

---

## 📚 참고 문서

- [README.md](prompt_discovery/README.md) - 시스템 상세 설명
- [config.yaml](config.yaml) - 설정 파일
- [verifier.py](verifier.py) - 메인 검증 시스템

---

## 💡 팁

1. **빠른 테스트**: 먼저 1~2개 클래스로 테스트
2. **GT 검증**: 테스트 DB를 기존 시스템으로 먼저 검증
3. **샘플 수**: 20~30개가 적당 (정확도 vs 시간)
4. **정기 업데이트**: 월 1회 정도 재최적화
5. **A/B 테스트**: 반드시 실제 데이터로 검증

---

완료! 🎉
