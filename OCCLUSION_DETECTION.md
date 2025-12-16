# 폐색된 사람 감지 개선 (Occluded Person Detection)

## 문제 설명

기존 시스템에서는 철조망이나 다른 물체에 부분적으로 가려진 사람을 철조망으로 잘못 분류하는 문제가 있었습니다.

## 해결 방법

### 1. 폐색 인식 프롬프트 추가

Person 클래스를 위한 특수 프롬프트를 추가했습니다:

**기존:**
```
"Is this object a person? Answer only 'Yes' or 'No'."
```

**개선:**
```
"Is there a person visible in this image, even if partially hidden or behind objects like fences or wires? Answer only 'Yes' or 'No'."
```

### 2. 동적 신뢰도 임계값

폐색이 예상되는 클래스(person, people, human, pedestrian)에 대해 낮은 신뢰도 임계값을 적용:

- **일반 객체**: confidence > 0.6 (기본값)
- **폐색 가능 객체**: confidence > 0.5 (완화된 값)

### 3. 설정 추가

`config.yaml`에 새로운 폐색 처리 설정이 추가되었습니다:

```yaml
verification:
  occlusion_handling:
    enabled: true  # 폐색 인식 기능 활성화
    use_context_prompt: true  # 컨텍스트 인식 프롬프트 사용
    confidence_threshold_occluded: 0.5  # 폐색된 객체의 신뢰도 임계값
    classes_with_occlusion: ["person", "people", "human", "pedestrian"]
```

## 변경된 파일

### Verifier (YOLO 라벨 검증)
1. **config.yaml** - 폐색 처리 설정 추가
2. **prompt_discovery/templates/current.yaml** - 폐색 인식 프롬프트 추가
3. **verifier.py** - 폐색 처리 로직 구현:
   - `__init__`: 폐색 설정 로드
   - `verify_single_box`: 동적 신뢰도 임계값 적용
   - `_load_optimized_prompts`: 프롬프트 로딩 개선

### Classifier (크롭 이미지 분류)
4. **classifier_config.yaml** - 폐색 처리 설정 추가
5. **image_classifier.py** - 폐색 처리 로직 구현:
   - `__init__`: 폐색 설정 로드
   - `_query_vlm`: 폐색 인식 프롬프트 사용
   - `classify_image`: person 감지 시 낮은 신뢰도 임계값 적용

## 사용 방법

### Verifier 설정 (config.yaml)

#### 활성화 (기본값: 켜짐)

```yaml
verification:
  occlusion_handling:
    enabled: true
```

#### 비활성화 (기존 방식 사용)

```yaml
verification:
  occlusion_handling:
    enabled: false
```

#### 신뢰도 임계값 조정

더 많은 폐색 케이스를 잡으려면:
```yaml
confidence_threshold_occluded: 0.4  # 더 낮게 (더 많이 잡음)
```

더 정확한 감지를 원하면:
```yaml
confidence_threshold_occluded: 0.6  # 더 높게 (더 정확)
```

#### 다른 클래스 추가

다른 클래스도 폐색 처리가 필요하면:
```yaml
classes_with_occlusion: ["person", "people", "human", "pedestrian", "car", "vehicle"]
```

### Classifier 설정 (classifier_config.yaml)

#### 활성화 (기본값: 켜짐)

```yaml
vlm:
  occlusion_handling:
    enabled: true
    occlusion_aware_prompt: "What is the main object in this image? If you see any part of a person, even if partially hidden behind objects like fences, wires, or other obstacles, answer 'person'. Otherwise, answer with one or two words only."
    confidence_threshold_occluded: 0.4
```

#### 비활성화 (기존 방식 사용)

```yaml
vlm:
  occlusion_handling:
    enabled: false
```

#### 프롬프트 커스터마이징

다른 프롬프트를 사용하려면:
```yaml
vlm:
  occlusion_handling:
    occlusion_aware_prompt: "당신의 커스텀 프롬프트"
```

## 작동 방식

1. **프롬프트 로딩**: 시스템 시작 시 `current.yaml`에서 폐색 인식 프롬프트 로드
2. **클래스 확인**: 각 박스 검증 시 해당 클래스가 폐색 클래스인지 확인
3. **임계값 적용**: 폐색 클래스면 낮은 신뢰도 임계값 사용
4. **검증**: VLM이 폐색을 고려한 프롬프트로 검증 수행

## 로그 예시

### Verifier 로그

폐색 처리가 활성화되면 다음과 같은 로그가 표시됩니다:

```
✓ Occlusion handling enabled for classes: {'person', 'people', 'human', 'pedestrian'}
  - Standard confidence threshold: 0.6
  - Occluded objects threshold: 0.5
✓ Loaded 47 optimized prompts
```

검증 시:
```
Using occlusion threshold 0.5 for person (standard: 0.6)
```

### Classifier 로그

폐색 처리가 활성화되면:

```
✓ Occlusion-aware classification enabled
  - Using specialized prompt for partially hidden persons
  - Occluded person confidence threshold: 0.4
```

분류 시:
```
person_image.jpg: Using occlusion threshold 0.4 for person (standard: 0.5)
```

## 테스트 방법

1. 철조망에 가려진 사람이 있는 이미지로 테스트
2. 기존 방식과 비교:
   - `enabled: false` → 철조망으로 잘못 분류
   - `enabled: true` → 사람으로 올바르게 인식
3. 로그에서 신뢰도 점수 확인

## 성능 영향

- **처리 속도**: 영향 없음 (동일한 VLM 호출 횟수)
- **메모리**: 영향 없음
- **정확도**: 폐색된 사람 감지율 향상

## 향후 개선 가능 사항

1. **다단계 검증**: 애매한 경우 추가 질문
2. **폐색 정도 감지**: 가려진 정도를 메타데이터로 저장
3. **컨텍스트 확장**: 주변 영역도 함께 분석
4. **클래스별 최적화**: 각 클래스마다 다른 임계값 사용
