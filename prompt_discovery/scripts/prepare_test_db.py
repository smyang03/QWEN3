#!/usr/bin/env python3
"""
Prepare Test DB Script
프로덕션 데이터에서 테스트 DB를 준비
"""

import sys
import yaml
import shutil
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_yolo_label(label_path: Path) -> List[int]:
    """YOLO 라벨 파일에서 클래스 ID 목록 추출"""
    class_ids = []

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.append(int(parts[0]))
    except Exception as e:
        logging.error(f"Failed to parse {label_path}: {e}")

    return class_ids


def collect_samples_by_class(
    images_dir: Path,
    labels_dir: Path
) -> Dict[int, List[Path]]:
    """클래스별로 이미지 파일 수집

    Returns:
        {class_id: [image_path, ...]}
    """
    class_samples = defaultdict(list)

    image_files = list(images_dir.glob('*.[jp][pn][g]'))
    logging.info(f"Found {len(image_files)} images")

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        # 라벨에서 클래스 추출
        class_ids = parse_yolo_label(label_path)

        for class_id in class_ids:
            class_samples[class_id].append(image_path)

    return class_samples


def prepare_test_db(
    source_images: Path,
    source_labels: Path,
    output_dir: Path,
    samples_per_class: int = 20,
    target_classes: List[int] = None,
    seed: int = 42
):
    """테스트 DB 준비

    Args:
        source_images: 원본 이미지 디렉토리
        source_labels: 원본 라벨 디렉토리
        output_dir: 출력 디렉토리
        samples_per_class: 클래스당 샘플 수
        target_classes: 대상 클래스 ID (None이면 모두)
        seed: 랜덤 시드
    """
    random.seed(seed)

    # 출력 디렉토리 생성 (YOLO 표준 구조)
    output_images = output_dir / 'JPEGImages'
    output_labels = output_dir / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    logging.info(f"Collecting samples from {source_images}...")

    # 클래스별 샘플 수집
    class_samples = collect_samples_by_class(source_images, source_labels)

    logging.info(f"Found {len(class_samples)} classes")

    # 클래스별 샘플 복사
    copied_count = 0
    selected_samples = {}

    for class_id, image_paths in sorted(class_samples.items()):
        if target_classes and class_id not in target_classes:
            continue

        # 랜덤 샘플링
        available = len(image_paths)
        sample_count = min(samples_per_class, available)

        selected = random.sample(image_paths, sample_count)
        selected_samples[class_id] = selected

        logging.info(f"Class {class_id}: selected {sample_count}/{available} samples")

        # 파일 복사
        for image_path in selected:
            label_path = source_labels / f"{image_path.stem}.txt"

            # 이미지 복사
            shutil.copy2(image_path, output_images / image_path.name)

            # 라벨 복사
            if label_path.exists():
                shutil.copy2(label_path, output_labels / label_path.name)

            copied_count += 1

    logging.info(f"\n✓ Copied {copied_count} samples to {output_dir}")

    # 요약 저장
    summary = {
        'source_images': str(source_images),
        'source_labels': str(source_labels),
        'samples_per_class': samples_per_class,
        'seed': seed,
        'classes': {
            int(class_id): {
                'sample_count': len(paths),
                'images': [p.name for p in paths]
            }
            for class_id, paths in selected_samples.items()
        }
    }

    summary_file = output_dir / 'summary.yaml'
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)

    logging.info(f"✓ Summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare test DB for prompt optimization')
    parser.add_argument('--source-images', type=str, help='Source images directory')
    parser.add_argument('--source-labels', type=str, help='Source labels directory')
    parser.add_argument('--output', type=str, default='prompt_discovery/test_db', help='Output directory')
    parser.add_argument('--samples', type=int, default=20, help='Samples per class')
    parser.add_argument('--classes', nargs='+', type=int, help='Target class IDs (default: all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file (for paths)')

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 경로 결정
    if args.source_images and args.source_labels:
        source_images = Path(args.source_images)
        source_labels = Path(args.source_labels)
    else:
        # config.yaml에서 경로 로드
        config_path = Path(args.config)
        if not config_path.exists():
            logging.error(f"Config file not found: {config_path}")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        source_images = Path(config['paths']['input_images'])
        source_labels = Path(config['paths']['input_labels'])

    if not source_images.exists():
        logging.error(f"Source images not found: {source_images}")
        return

    if not source_labels.exists():
        logging.error(f"Source labels not found: {source_labels}")
        return

    output_dir = Path(args.output)

    logging.info(f"Source images: {source_images}")
    logging.info(f"Source labels: {source_labels}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Samples per class: {args.samples}")

    # 테스트 DB 준비
    prepare_test_db(
        source_images=source_images,
        source_labels=source_labels,
        output_dir=output_dir,
        samples_per_class=args.samples,
        target_classes=args.classes,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
