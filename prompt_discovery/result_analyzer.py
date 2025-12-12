#!/usr/bin/env python3
"""
Result Analyzer
프롬프트 테스트 결과를 분석하고 최적 프롬프트 저장
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

from prompt_tester import PromptScore


class ResultAnalyzer:
    """결과 분석 및 저장"""

    def __init__(self, config: Dict):
        self.config = config

        # 결과 저장 경로
        self.results_dir = Path('prompt_discovery/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 템플릿 저장 경로
        self.templates_dir = Path('prompt_discovery/templates')
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def analyze_and_save(
        self,
        class_name: str,
        scores: List[PromptScore],
        save_top_n: int = 10
    ) -> Dict:
        """결과 분석 및 저장

        Args:
            class_name: 클래스 이름
            scores: 테스트 점수 리스트 (정확도 순으로 정렬됨)
            save_top_n: 상위 N개 결과 저장

        Returns:
            분석 결과 딕셔너리
        """
        if not scores:
            logging.warning(f"No scores to analyze for class '{class_name}'")
            return {}

        # 최고 점수
        best_score = scores[0]

        # 상위 N개
        top_scores = scores[:save_top_n]

        # 분석 결과
        analysis = {
            'class_name': class_name,
            'timestamp': datetime.now().isoformat(),
            'total_prompts_tested': len(scores),
            'best_prompt': {
                'prompt': best_score.prompt,
                'accuracy': best_score.accuracy,
                'avg_confidence': best_score.avg_confidence,
                'correct_count': best_score.correct_count,
                'total_count': best_score.total_count
            },
            'top_prompts': [
                {
                    'rank': i + 1,
                    'prompt': score.prompt,
                    'accuracy': score.accuracy,
                    'avg_confidence': score.avg_confidence,
                    'correct_count': score.correct_count,
                    'total_count': score.total_count
                }
                for i, score in enumerate(top_scores)
            ],
            'statistics': {
                'best_accuracy': best_score.accuracy,
                'worst_accuracy': scores[-1].accuracy,
                'avg_accuracy': sum(s.accuracy for s in scores) / len(scores),
                'median_accuracy': scores[len(scores) // 2].accuracy if scores else 0.0
            }
        }

        # JSON 저장
        result_file = self.results_dir / f"{class_name}_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        logging.info(f"✓ Results saved: {result_file}")

        # 상세 결과 저장 (모든 샘플별 결과)
        detailed_file = self.results_dir / f"{class_name}_detailed.json"
        detailed_results = []

        for score in top_scores:
            detailed_results.append({
                'prompt': score.prompt,
                'accuracy': score.accuracy,
                'sample_results': [
                    {
                        'sample_id': r.sample_id,
                        'is_correct': r.is_correct,
                        'confidence': r.confidence,
                        'response': r.response
                    }
                    for r in score.results
                ]
            })

        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logging.info(f"✓ Detailed results saved: {detailed_file}")

        return analysis

    def save_best_prompt_template(self, class_results: Dict[str, Dict]) -> Path:
        """여러 클래스의 최적 프롬프트를 템플릿으로 저장

        Args:
            class_results: {class_name: analysis_result}

        Returns:
            저장된 템플릿 파일 경로
        """
        # 템플릿 생성
        template = {
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'created_at': datetime.now().isoformat(),
            'prompts': {}
        }

        for class_name, analysis in class_results.items():
            if 'best_prompt' in analysis:
                template['prompts'][class_name] = {
                    'prompt': analysis['best_prompt']['prompt'],
                    'accuracy': analysis['best_prompt']['accuracy'],
                    'avg_confidence': analysis['best_prompt']['avg_confidence']
                }

        # 템플릿 저장
        template_file = self.templates_dir / f"optimized_{template['version']}.yaml"
        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, allow_unicode=True, default_flow_style=False)

        logging.info(f"✓ Template saved: {template_file}")

        # current.yaml 심볼릭 링크 생성
        current_link = self.templates_dir / 'current.yaml'
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()

        current_link.symlink_to(template_file.name)
        logging.info(f"✓ Current template updated: {current_link}")

        return template_file

    def load_template(self, template_name: str = 'current.yaml') -> Dict:
        """템플릿 로드

        Args:
            template_name: 템플릿 파일명

        Returns:
            템플릿 딕셔너리
        """
        template_file = self.templates_dir / template_name

        if not template_file.exists():
            logging.warning(f"Template not found: {template_file}")
            return {}

        with open(template_file, 'r', encoding='utf-8') as f:
            template = yaml.safe_load(f)

        return template

    def get_prompt_for_class(self, class_name: str, template_name: str = 'current.yaml') -> str:
        """특정 클래스의 최적 프롬프트 가져오기

        Args:
            class_name: 클래스 이름
            template_name: 템플릿 파일명

        Returns:
            프롬프트 문자열 (없으면 기본 프롬프트)
        """
        template = self.load_template(template_name)

        if 'prompts' in template and class_name in template['prompts']:
            return template['prompts'][class_name]['prompt']
        else:
            # Fallback
            fallback = f"Is this object a {class_name}? Answer Yes or No."
            logging.warning(f"No optimized prompt for '{class_name}', using fallback")
            return fallback

    def print_summary(self, class_results: Dict[str, Dict]):
        """결과 요약 출력

        Args:
            class_results: {class_name: analysis_result}
        """
        print("\n" + "=" * 80)
        print("Prompt Optimization Results Summary")
        print("=" * 80)

        for class_name, analysis in class_results.items():
            if 'best_prompt' not in analysis:
                continue

            best = analysis['best_prompt']
            print(f"\n[{class_name}]")
            print(f"  Best Accuracy: {best['accuracy']:.3f} ({best['correct_count']}/{best['total_count']})")
            print(f"  Avg Confidence: {best['avg_confidence']:.3f}")
            print(f"  Prompt: {best['prompt'][:70]}...")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # 테스트 코드
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    analyzer = ResultAnalyzer(config)

    # 더미 결과로 테스트
    from prompt_tester import PromptScore, TestResult

    dummy_results = [
        TestResult(
            prompt="Test prompt 1",
            sample_id=i,
            class_name="helmet",
            is_correct=(i % 2 == 0),
            confidence=0.8 + (i % 3) * 0.05,
            response="Yes" if (i % 2 == 0) else "No",
            gt_label=True
        )
        for i in range(10)
    ]

    dummy_score = PromptScore(
        prompt="Test prompt 1",
        accuracy=0.5,
        avg_confidence=0.85,
        correct_count=5,
        total_count=10,
        results=dummy_results
    )

    # 분석 및 저장
    analysis = analyzer.analyze_and_save("helmet", [dummy_score])

    # 템플릿 저장
    class_results = {"helmet": analysis}
    template_file = analyzer.save_best_prompt_template(class_results)

    # 요약 출력
    analyzer.print_summary(class_results)

    # 프롬프트 로드 테스트
    prompt = analyzer.get_prompt_for_class("helmet")
    print(f"\nLoaded prompt for 'helmet': {prompt}")
