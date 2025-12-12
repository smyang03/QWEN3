#!/usr/bin/env python3
"""
Prompt Optimization Main Script
클래스별 최적 프롬프트를 찾아서 저장
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# 오프라인 모드 설정 (Hugging Face)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# 상위 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import verifier
from gt_loader import GroundTruthLoader
from prompt_generator import PromptGenerator
from prompt_tester import PromptTester
from result_analyzer import ResultAnalyzer


def optimize_single_class(
    class_name: str,
    config: Dict,
    model_manager,
    show_progress: bool = True
) -> Dict:
    """단일 클래스의 프롬프트 최적화

    Args:
        class_name: 클래스 이름
        config: 설정
        model_manager: 모델 관리자
        show_progress: 진행률 표시

    Returns:
        분석 결과 딕셔너리
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Optimizing prompts for class: {class_name}")
    logging.info(f"{'='*80}\n")

    # 1. GT 샘플 로드
    loader = GroundTruthLoader(config)
    samples = loader.load_by_class(class_name)

    if not samples:
        logging.error(f"No GT samples found for class '{class_name}'")
        return {}

    min_samples = config.get('prompt_optimization', {}).get('ground_truth', {}).get('min_samples_per_class', 10)
    if len(samples) < min_samples:
        logging.warning(f"Only {len(samples)} samples found (minimum: {min_samples})")

    logging.info(f"Loaded {len(samples)} GT samples")

    # 2. 프롬프트 후보 생성
    generator = PromptGenerator(config)
    prompts = generator.generate_candidates(class_name)

    max_candidates = config.get('prompt_optimization', {}).get('testing', {}).get('max_candidates', 30)
    prompts = prompts[:max_candidates]

    logging.info(f"Generated {len(prompts)} prompt candidates")

    # 3. 프롬프트 테스트
    tester = PromptTester(model_manager, config)
    scores = tester.test_all_prompts(prompts, samples, show_progress=show_progress)

    # 4. 결과 분석 및 저장
    analyzer = ResultAnalyzer(config)
    save_top_n = config.get('prompt_optimization', {}).get('testing', {}).get('save_top_n', 10)
    analysis = analyzer.analyze_and_save(class_name, scores, save_top_n=save_top_n)

    return analysis


def optimize_multiple_classes(
    class_names: List[str],
    config: Dict,
    model_manager,
    show_progress: bool = True
) -> Dict[str, Dict]:
    """여러 클래스의 프롬프트 최적화

    Args:
        class_names: 클래스 이름 리스트
        config: 설정
        model_manager: 모델 관리자
        show_progress: 진행률 표시

    Returns:
        {class_name: analysis_result}
    """
    results = {}

    for i, class_name in enumerate(class_names, 1):
        logging.info(f"\n\nProcessing class {i}/{len(class_names)}: {class_name}\n")

        try:
            analysis = optimize_single_class(class_name, config, model_manager, show_progress)
            results[class_name] = analysis

        except Exception as e:
            logging.error(f"Failed to optimize class '{class_name}': {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description='Optimize VLM prompts for YOLO classes')
    parser.add_argument('--class', dest='class_name', type=str, help='Single class name to optimize')
    parser.add_argument('--all', action='store_true', help='Optimize all classes')
    parser.add_argument('--classes', nargs='+', help='List of class names to optimize')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 로드
    logging.info("Loading VLM model...")
    model_manager = verifier.ModelManager(config)
    model_manager.load_model()

    # 클래스 결정
    if args.class_name:
        class_names = [args.class_name]
    elif args.classes:
        class_names = args.classes
    elif args.all:
        # GT에서 사용 가능한 클래스 자동 탐지
        loader = GroundTruthLoader(config)
        distribution = loader.get_class_distribution()
        class_names = list(distribution.keys())
        logging.info(f"Found {len(class_names)} classes in GT: {class_names}")
    else:
        logging.error("Please specify --class, --classes, or --all")
        parser.print_help()
        return

    show_progress = not args.no_progress

    # 최적화 실행
    if len(class_names) == 1:
        results = {class_names[0]: optimize_single_class(class_names[0], config, model_manager, show_progress)}
    else:
        results = optimize_multiple_classes(class_names, config, model_manager, show_progress)

    # 템플릿 저장
    analyzer = ResultAnalyzer(config)
    template_file = analyzer.save_best_prompt_template(results)

    # 요약 출력
    analyzer.print_summary(results)

    logging.info(f"\n✓ Optimization complete!")
    logging.info(f"✓ Template saved: {template_file}")
    logging.info(f"\nTo use optimized prompts, set in config.yaml:")
    logging.info(f"  prompt_optimization:")
    logging.info(f"    production:")
    logging.info(f"      use_optimized_prompts: true")


if __name__ == "__main__":
    main()
