# 폴더 구조 단순화 변경 사항

출력 폴더 구조를 복잡한 3단계에서 간단한 1단계로 변경했습니다.

---

## 📂 변경 전 (복잡함)

```
output/
├── correct/
│   ├── indoor/
│   │   ├── day/
│   │   │   ├── JPEGImages/
│   │   │   └── labels/
│   │   └── night/
│   │       ├── JPEGImages/
│   │       └── labels/
│   └── outdoor/
│       ├── day/
│       └── night/
├── mislabeled/
│   ├── indoor/
│   │   ├── day/
│   │   └── night/
│   └── outdoor/
│       ├── day/
│       └── night/
├── uncertain/
│   └── ...
└── debug_images/
```

**문제점:**
- category/location/time 3단계로 너무 깊음
- 파일 찾기 어려움
- 폴더가 너무 많음 (3 × 3 × 3 = 27개 가능)

---

## 📂 변경 후 (간단함)

```
output/
├── correct/
│   ├── JPEGImages/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── mislabeled/
│   ├── JPEGImages/
│   ├── labels/              # 수정된 라벨
│   └── labels_original/     # 원본 백업
├── uncertain/
│   ├── JPEGImages/
│   └── labels/
└── debug_images/
    ├── image1_verified.jpg
    └── ...
```

**개선점:**
- category별로만 구분
- 간단하고 직관적
- 3개 폴더만 (correct, mislabeled, uncertain)
- 파일 찾기 쉬움

---

## 🔄 변경된 코드

### 1. verifier.py - ResultManager

#### copy_files()
**Before:**
```python
def copy_files(self, image_path: Path, label_path: Path, 
               category: str, scene_info: SceneInfo):
    target_folder = self.output_base / category / scene_info.location / scene_info.time
```

**After:**
```python
def copy_files(self, image_path: Path, label_path: Path, 
               category: str, scene_info: SceneInfo = None):
    target_folder = self.output_base / category
```

#### save_corrected_label()
**Before:**
```python
def save_corrected_label(..., scene_info: SceneInfo):
    target_folder = self.output_base / category / scene_info.location / scene_info.time
```

**After:**
```python
def save_corrected_label(..., scene_info: SceneInfo = None):
    target_folder = self.output_base / category
```

---

### 2. 멀티 GPU 스크립트

#### run_multigpu_nocopy.sh / run_multigpu_simple.sh

**Before:**
```python
for category in categories:
    for location in locations:
        for time in times:
            src_folder = gpu_output / category / location / time
```

**After:**
```python
for category in categories:
    src_folder = gpu_output / category
```

---

## 💡 Scene 정보는 어떻게?

**Scene 분류는 계속 작동합니다:**
- Scene 정보(indoor/outdoor, day/night)는 여전히 수집
- JSON 리포트에 저장
- 통계에 포함

**단지 폴더 구조에만 반영하지 않습니다:**
- 파일은 category별로만 구분
- Scene 정보는 리포트로 확인 가능

---

## 📊 리포트에서 Scene 정보 확인

### verification_report.json
```json
{
  "statistics": {
    "total_images": 258180,
    "correct": 200000,
    "mislabeled": 50000,
    "uncertain": 8180,
    "per_scene": {
      "outdoor": {
        "day": 150000,
        "night": 50000
      },
      "indoor": {
        "day": 40000,
        "night": 18180
      }
    }
  },
  "detailed_results": [
    {
      "image": "image1.jpg",
      "category": "correct",
      "scene_location": "outdoor",
      "scene_time": "day",
      ...
    }
  ]
}
```

---

## 📋 사용 예시

### 간단한 파일 접근
```bash
# Correct 이미지 확인
ls output/correct/JPEGImages/

# Mislabeled 원본 라벨
ls output/mislabeled/labels_original/

# Mislabeled 수정된 라벨
ls output/mislabeled/labels/

# 디버그 이미지
ls output/debug_images/
```

### Scene별 필터링 (필요 시)
```bash
# Scene 정보로 필터링하려면 JSON 리포트 사용
python3 << EOF
import json
from pathlib import Path

with open('output/verification_report.json') as f:
    data = json.load(f)

# Outdoor + Day인 이미지만 추출
outdoor_day = [
    r['image'] for r in data['detailed_results']
    if r['scene_location'] == 'outdoor' and r['scene_time'] == 'day'
]

print(f"Outdoor/Day images: {len(outdoor_day)}")
for img in outdoor_day[:10]:
    print(f"  {img}")
EOF
```

---

## 🎯 장점

### 1. 단순함
- 3단계 → 1단계
- 파일 찾기 쉬움
- 폴더 구조 이해 쉬움

### 2. 빠른 접근
```bash
# Before (복잡)
cd output/mislabeled/outdoor/day/JPEGImages/

# After (간단)
cd output/mislabeled/JPEGImages/
```

### 3. 도구 호환성
```bash
# YOLO 학습 데이터셋으로 바로 사용
# 폴더 구조가 표준 YOLO 형식과 동일
output/correct/
├── JPEGImages/
└── labels/
```

### 4. 디스크 효율
- 중첩 폴더 감소
- 메타데이터 오버헤드 감소

---

## 🔄 기존 데이터 마이그레이션

기존 복잡한 구조의 데이터를 단순한 구조로 변환:

```bash
#!/bin/bash
# migrate_structure.sh

OLD_OUTPUT="output_old"
NEW_OUTPUT="output_new"

mkdir -p "$NEW_OUTPUT"/{correct,mislabeled,uncertain}/{JPEGImages,labels}
mkdir -p "$NEW_OUTPUT/mislabeled/labels_original"

# Correct 이동
find "$OLD_OUTPUT/correct" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | \
    xargs -I{} cp {} "$NEW_OUTPUT/correct/JPEGImages/"

find "$OLD_OUTPUT/correct" -name "*.txt" | \
    xargs -I{} cp {} "$NEW_OUTPUT/correct/labels/"

# Mislabeled 이동
find "$OLD_OUTPUT/mislabeled" -path "*/JPEGImages/*" \( -name "*.jpg" -o -name "*.png" \) | \
    xargs -I{} cp {} "$NEW_OUTPUT/mislabeled/JPEGImages/"

find "$OLD_OUTPUT/mislabeled" -path "*/labels/*" -name "*.txt" | \
    xargs -I{} cp {} "$NEW_OUTPUT/mislabeled/labels/"

find "$OLD_OUTPUT/mislabeled" -path "*/labels_original/*" -name "*.txt" | \
    xargs -I{} cp {} "$NEW_OUTPUT/mislabeled/labels_original/"

# Uncertain 이동
find "$OLD_OUTPUT/uncertain" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | \
    xargs -I{} cp {} "$NEW_OUTPUT/uncertain/JPEGImages/"

find "$OLD_OUTPUT/uncertain" -name "*.txt" | \
    xargs -I{} cp {} "$NEW_OUTPUT/uncertain/labels/"

echo "Migration complete!"
```

---

## ⚠️ 주의사항

### 1. Scene 정보가 필요한 경우
Scene별로 데이터를 구분해야 한다면:
- JSON 리포트 참조
- 또는 custom 스크립트로 재분류

### 2. 기존 스크립트
폴더 구조에 의존하는 custom 스크립트는 수정 필요

### 3. 하위 호환성
이전 버전으로 돌아가려면:
- `verifier.py`를 이전 버전으로 복원
- 또는 config에서 scene 분류 비활성화

---

## 🎨 시각적 비교

### Before
```
output/
  correct/
    indoor/
      day/      ← 3단계 깊이
    outdoor/
      night/    ← 찾기 어려움
```

### After
```
output/
  correct/    ← 1단계 깊이
  mislabeled/ ← 바로 접근
  uncertain/  ← 간단함
```

---

## ✅ 체크리스트

수정된 버전 확인:

- [x] verifier.py - copy_files() 수정
- [x] verifier.py - save_corrected_label() 수정
- [x] run_multigpu_nocopy.sh - merge 부분 수정
- [x] run_multigpu_simple.sh - merge 부분 수정
- [x] Scene 정보는 JSON 리포트에 유지
- [x] 통계 수집은 계속 작동

---

## 🚀 즉시 사용

```bash
# 단일 실행
python verifier.py

# 멀티 GPU 실행
bash run_multigpu_nocopy.sh

# 결과 확인
ls -la output/correct/JPEGImages/
ls -la output/mislabeled/labels/
```

---

**이제 간단하고 직관적인 구조로 작업할 수 있습니다!** 🎉
