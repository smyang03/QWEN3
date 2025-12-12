#!/usr/bin/env python3
"""
Prompt Generator
다양한 프롬프트 후보를 생성
"""

from typing import List, Dict
import logging


class PromptGenerator:
    """프롬프트 후보 생성기"""

    def __init__(self, config: Dict):
        self.config = config

    def generate_candidates(self, class_name: str) -> List[str]:
        """클래스별 프롬프트 후보 생성

        Args:
            class_name: 클래스 이름 (예: "helmet", "person")

        Returns:
            프롬프트 후보 리스트
        """
        candidates = []

        # 1. 기본 질문 형식
        candidates.extend(self._generate_basic_prompts(class_name))

        # 2. 디테일 강조 형식
        candidates.extend(self._generate_detailed_prompts(class_name))

        # 3. 컨텍스트 포함 형식
        candidates.extend(self._generate_contextual_prompts(class_name))

        # 4. 부정 질문 형식
        candidates.extend(self._generate_negative_prompts(class_name))

        # 5. 설명 요청 형식
        candidates.extend(self._generate_descriptive_prompts(class_name))

        # 중복 제거
        candidates = list(dict.fromkeys(candidates))

        logging.info(f"Generated {len(candidates)} prompt candidates for '{class_name}'")
        return candidates

    def _generate_basic_prompts(self, class_name: str) -> List[str]:
        """기본 질문 형식"""
        return [
            f"Is this a {class_name}? Answer Yes or No.",
            f"Is there a {class_name} in this image? Answer Yes or No.",
            f"Does this image contain a {class_name}? Answer Yes or No.",
            f"Can you see a {class_name}? Answer Yes or No.",
            f"Is this object a {class_name}? Answer Yes or No.",
        ]

    def _generate_detailed_prompts(self, class_name: str) -> List[str]:
        """디테일 강조 형식"""
        return [
            f"Looking at this image carefully, is this a {class_name}? Answer Yes or No.",
            f"Based on the visual features, is this a {class_name}? Answer Yes or No.",
            f"Examining the details, is this object a {class_name}? Answer Yes or No.",
            f"Considering the shape and appearance, is this a {class_name}? Answer Yes or No.",
            f"After careful observation, is this a {class_name}? Answer Yes or No.",
        ]

    def _generate_contextual_prompts(self, class_name: str) -> List[str]:
        """컨텍스트 포함 형식"""
        # 클래스별 컨텍스트 힌트
        context_hints = {
            'helmet': 'safety equipment',
            'person': 'human being',
            'car': 'vehicle',
            'forklift': 'industrial equipment',
            'slip': 'accident or fall',
            'head': 'body part',
            'chair': 'furniture',
            'bird': 'animal',
            'truck': 'vehicle',
            'bus': 'vehicle',
        }

        hint = context_hints.get(class_name, 'object')

        return [
            f"Is this {class_name} ({hint}) present? Answer Yes or No.",
            f"Looking at this as {hint}, is it a {class_name}? Answer Yes or No.",
            f"In the context of {hint}, is this a {class_name}? Answer Yes or No.",
        ]

    def _generate_negative_prompts(self, class_name: str) -> List[str]:
        """부정 질문 형식"""
        return [
            f"Is this NOT a {class_name}? Answer Yes or No.",
            f"Could this be anything other than a {class_name}? Answer Yes or No.",
        ]

    def _generate_descriptive_prompts(self, class_name: str) -> List[str]:
        """설명 요청 형식"""
        return [
            f"Identify if this is a {class_name}. Reply only Yes or No.",
            f"Determine whether this is a {class_name}. Answer Yes or No.",
            f"Classify this object: is it a {class_name}? Answer Yes or No.",
            f"Verify if this is a {class_name}. Answer Yes or No.",
        ]

    def add_custom_candidates(self, class_name: str, custom_prompts: List[str]) -> List[str]:
        """사용자 정의 프롬프트 추가

        Args:
            class_name: 클래스 이름
            custom_prompts: 사용자가 추가하는 프롬프트 리스트

        Returns:
            기본 후보 + 사용자 정의 프롬프트
        """
        candidates = self.generate_candidates(class_name)
        candidates.extend(custom_prompts)

        # 중복 제거
        candidates = list(dict.fromkeys(candidates))

        logging.info(f"Added {len(custom_prompts)} custom prompts. Total: {len(candidates)}")
        return candidates


if __name__ == "__main__":
    # 테스트 코드
    import yaml
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    generator = PromptGenerator(config)

    # 테스트 클래스
    test_classes = ['helmet', 'person', 'car']

    for class_name in test_classes:
        print(f"\n=== Prompt candidates for '{class_name}' ===")
        candidates = generator.generate_candidates(class_name)
        for i, prompt in enumerate(candidates, 1):
            print(f"  [{i:2d}] {prompt}")

    # 커스텀 프롬프트 추가 테스트
    print(f"\n=== Custom prompts test ===")
    custom = [
        "Is this a safety helmet worn on construction sites? Yes/No",
        "Hard hat detected? Yes/No"
    ]
    all_prompts = generator.add_custom_candidates('helmet', custom)
    print(f"Total prompts: {len(all_prompts)}")
