#!/usr/bin/env python3
"""
Prompt Tester
각 프롬프트 후보를 GT 샘플로 테스트하고 정확도 측정
"""

import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

from gt_loader import GTSample


@dataclass
class TestResult:
    """단일 테스트 결과"""
    prompt: str
    sample_id: int
    class_name: str
    is_correct: bool
    confidence: float
    response: str
    gt_label: bool  # Ground Truth (항상 True, 해당 클래스 샘플이므로)


@dataclass
class PromptScore:
    """프롬프트 점수"""
    prompt: str
    accuracy: float
    avg_confidence: float
    correct_count: int
    total_count: int
    results: List[TestResult]


class PromptTester:
    """프롬프트 테스터"""

    def __init__(self, model_manager, config: Dict):
        """
        Args:
            model_manager: verifier.ModelManager 인스턴스
            config: 전체 설정
        """
        self.model_manager = model_manager
        self.config = config

        # 모델 로드 확인
        if self.model_manager.model is None or self.model_manager.processor is None:
            logging.info("Loading model in PromptTester...")
            self.model_manager.load_model()

        # Confidence threshold
        self.confidence_threshold = config.get('verification', {}).get('confidence_threshold', 0.6)

    def test_single_prompt(
        self,
        prompt: str,
        samples: List[GTSample],
        show_progress: bool = True
    ) -> PromptScore:
        """단일 프롬프트를 모든 샘플로 테스트

        Args:
            prompt: 테스트할 프롬프트
            samples: GT 샘플 리스트
            show_progress: 진행률 표시 여부

        Returns:
            PromptScore
        """
        results = []

        iterator = tqdm(samples, desc=f"Testing prompt", disable=not show_progress)

        for idx, sample in enumerate(iterator):
            # PIL 이미지로 변환
            pil_image = Image.fromarray(sample.crop_image)

            # 크기 조정
            max_size = self.config.get('verification', {}).get('max_image_size', 1280)
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # 모델 쿼리
            is_correct, confidence, response = self._query_model(pil_image, prompt)

            # 결과 저장
            result = TestResult(
                prompt=prompt,
                sample_id=idx,
                class_name=sample.class_name,
                is_correct=is_correct,
                confidence=confidence,
                response=response,
                gt_label=True  # GT 샘플이므로 항상 True
            )
            results.append(result)

        # 정확도 계산
        correct_count = sum(1 for r in results if r.is_correct)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # 평균 confidence
        avg_confidence = sum(r.confidence for r in results) / total_count if total_count > 0 else 0.0

        score = PromptScore(
            prompt=prompt,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            correct_count=correct_count,
            total_count=total_count,
            results=results
        )

        return score

    def test_all_prompts(
        self,
        prompts: List[str],
        samples: List[GTSample],
        show_progress: bool = True
    ) -> List[PromptScore]:
        """모든 프롬프트 후보를 테스트

        Args:
            prompts: 프롬프트 후보 리스트
            samples: GT 샘플 리스트
            show_progress: 진행률 표시 여부

        Returns:
            정확도 순으로 정렬된 PromptScore 리스트
        """
        scores = []

        logging.info(f"Testing {len(prompts)} prompts on {len(samples)} samples...")

        for i, prompt in enumerate(prompts, 1):
            logging.info(f"\n[{i}/{len(prompts)}] Testing: {prompt[:60]}...")

            score = self.test_single_prompt(prompt, samples, show_progress=show_progress)

            logging.info(f"  Accuracy: {score.accuracy:.3f} ({score.correct_count}/{score.total_count})")
            logging.info(f"  Avg Confidence: {score.avg_confidence:.3f}")

            scores.append(score)

            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 정확도 순으로 정렬 (같으면 confidence로 정렬)
        scores.sort(key=lambda x: (x.accuracy, x.avg_confidence), reverse=True)

        return scores

    def _query_model(self, image: Image.Image, prompt: str) -> Tuple[bool, float, str]:
        """모델에게 질문 (verifier.LabelVerifier의 로직 재사용)

        Args:
            image: PIL 이미지
            prompt: 프롬프트

        Returns:
            (is_correct, confidence, response)
        """
        try:
            # 메시지 구성
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 텍스트 생성
            text = self.model_manager.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 이미지 처리
            inputs = self.model_manager.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(self.model_manager.device)

            # 생성 (scores 포함)
            with torch.no_grad():
                outputs = self.model_manager.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.3,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            # 디코딩
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
            ]

            response = self.model_manager.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Yes/No 파싱
            is_correct = self._parse_yes_no(response)

            # Confidence 계산 (Logit 기반)
            confidence = self._compute_confidence(outputs, inputs)

            # Threshold 적용
            if confidence < self.confidence_threshold:
                is_correct = False  # 낮은 confidence는 오답 처리

            return is_correct, confidence, response

        except Exception as e:
            logging.error(f"Model query failed: {e}")
            return False, 0.0, f"ERROR: {str(e)}"

    def _parse_yes_no(self, response: str) -> bool:
        """응답에서 Yes/No 파싱"""
        response_lower = response.lower()

        has_yes = 'yes' in response_lower
        has_no = 'no' in response_lower

        if has_yes and not has_no:
            return True
        elif has_no and not has_yes:
            return False
        else:
            # 애매한 경우 False (오답)
            return False

    def _compute_confidence(self, outputs, inputs) -> float:
        """Confidence 계산 (Logit 기반)"""
        try:
            # Transition scores 계산
            transition_scores = self.model_manager.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )

            # Log 확률 → 실제 확률
            token_probs = torch.exp(transition_scores[0])

            # 평균 확률
            confidence = token_probs.mean().item()

            # 0~1 범위 보장
            confidence = max(0.0, min(1.0, confidence))

            return confidence

        except Exception as e:
            logging.warning(f"Failed to compute confidence: {e}")
            return 0.5


if __name__ == "__main__":
    # 테스트 코드
    import os
    import yaml
    from pathlib import Path
    import sys

    # 오프라인 모드 설정 (Hugging Face)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

    # verifier 모듈 임포트
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import verifier
    from gt_loader import GroundTruthLoader
    from prompt_generator import PromptGenerator

    logging.basicConfig(level=logging.INFO)

    # Config 로드
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 로드
    model_manager = verifier.ModelManager(config)
    model_manager.load_model()

    # GT 로드
    loader = GroundTruthLoader(config)
    samples = loader.load_by_class('helmet')

    if not samples:
        print("No samples found for 'helmet'")
        exit(1)

    # 프롬프트 생성
    generator = PromptGenerator(config)
    prompts = generator.generate_candidates('helmet')[:3]  # 테스트: 3개만

    # 테스트
    tester = PromptTester(model_manager, config)
    scores = tester.test_all_prompts(prompts, samples[:5])  # 테스트: 5개 샘플만

    # 결과 출력
    print("\n=== Test Results ===")
    for i, score in enumerate(scores, 1):
        print(f"\n[{i}] {score.prompt}")
        print(f"    Accuracy: {score.accuracy:.3f} ({score.correct_count}/{score.total_count})")
        print(f"    Avg Confidence: {score.avg_confidence:.3f}")
