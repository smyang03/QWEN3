#!/usr/bin/env python3
"""
Ground Truth Loader
테스트 DB를 로드하고 클래스별로 샘플을 분류
"""

import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class GTSample:
    """Ground Truth 샘플"""
    image_path: Path
    image: np.ndarray  # RGB
    crop_image: np.ndarray  # RGB, 크롭된 영역
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # x_center, y_center, width, height (normalized)
    bbox_pixel: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel)


class GroundTruthLoader:
    """Ground Truth 데이터 로더"""

    def __init__(self, config: Dict):
        self.config = config

        # 경로 설정
        gt_config = config.get('prompt_optimization', {}).get('ground_truth', {})
        self.gt_path = Path(gt_config.get('path', 'prompt_discovery/test_db'))
        self.images_dir = self.gt_path / 'images'
        self.labels_dir = self.gt_path / 'labels'

        # 클래스 맵 로드
        self.class_map = self._load_class_map()

        logging.info(f"GT Path: {self.gt_path}")
        logging.info(f"Classes: {list(self.class_map.values())}")

    def _load_class_map(self) -> Dict[int, str]:
        """클래스 맵 로드 (config.yaml의 classes 사용)"""
        raw_classes = self.config['classes']
        class_map = {}

        for class_id, names in raw_classes.items():
            class_id = int(class_id)

            # 리스트 또는 단일 문자열 지원
            if isinstance(names, list):
                primary_name = names[0]
            else:
                primary_name = names

            class_map[class_id] = primary_name

        return class_map

    def load_all(self) -> List[GTSample]:
        """모든 GT 샘플 로드"""
        samples = []

        if not self.images_dir.exists():
            logging.error(f"Images directory not found: {self.images_dir}")
            return samples

        if not self.labels_dir.exists():
            logging.error(f"Labels directory not found: {self.labels_dir}")
            return samples

        # 이미지 파일 목록
        image_files = sorted(self.images_dir.glob('*.[jp][pn][g]'))

        logging.info(f"Found {len(image_files)} images in GT")

        for image_path in image_files:
            # 라벨 파일 찾기
            label_path = self.labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                logging.warning(f"Label not found for {image_path.name}")
                continue

            # 이미지 로드 (한글 경로 지원)
            try:
                with open(image_path, 'rb') as f:
                    img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    logging.error(f"Failed to decode image: {image_path}")
                    continue

                # RGB 변환
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            except Exception as e:
                logging.error(f"Failed to load image {image_path}: {e}")
                continue

            # 라벨 파싱
            boxes = self._parse_label_file(label_path)

            # 각 박스를 샘플로 변환
            for box in boxes:
                class_id, x_center, y_center, width, height = box

                # 클래스 이름
                if class_id not in self.class_map:
                    logging.warning(f"Unknown class_id {class_id} in {image_path.name}")
                    continue

                class_name = self.class_map[class_id]

                # 픽셀 좌표 계산
                h, w = img_rgb.shape[:2]
                x_center_px = x_center * w
                y_center_px = y_center * h
                w_px = width * w
                h_px = height * h

                x1 = int(x_center_px - w_px / 2)
                y1 = int(y_center_px - h_px / 2)
                x2 = int(x_center_px + w_px / 2)
                y2 = int(y_center_px + h_px / 2)

                # 경계 체크
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # 크롭 (패딩 추가)
                padding = self.config.get('verification', {}).get('crop_padding', 10)
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x2 + padding)
                y2_pad = min(h, y2 + padding)

                crop_img = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad]

                # 샘플 생성
                sample = GTSample(
                    image_path=image_path,
                    image=img_rgb,
                    crop_image=crop_img,
                    class_id=class_id,
                    class_name=class_name,
                    bbox=(x_center, y_center, width, height),
                    bbox_pixel=(x1, y1, x2, y2)
                )

                samples.append(sample)

        logging.info(f"Loaded {len(samples)} GT samples")
        return samples

    def load_by_class(self, class_name: str) -> List[GTSample]:
        """특정 클래스의 샘플만 로드"""
        all_samples = self.load_all()

        class_samples = [s for s in all_samples if s.class_name == class_name]

        logging.info(f"Found {len(class_samples)} samples for class '{class_name}'")
        return class_samples

    def get_class_distribution(self) -> Dict[str, int]:
        """클래스별 샘플 개수 반환"""
        all_samples = self.load_all()

        distribution = {}
        for sample in all_samples:
            if sample.class_name not in distribution:
                distribution[sample.class_name] = 0
            distribution[sample.class_name] += 1

        return distribution

    def _parse_label_file(self, label_path: Path) -> List[Tuple]:
        """YOLO 라벨 파일 파싱"""
        boxes = []

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        boxes.append((class_id, x_center, y_center, width, height))

        except Exception as e:
            logging.error(f"Error parsing label file {label_path}: {e}")

        return boxes


if __name__ == "__main__":
    # 테스트 코드
    import sys
    logging.basicConfig(level=logging.INFO)

    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    loader = GroundTruthLoader(config)

    # 클래스 분포 출력
    print("\n=== Class Distribution ===")
    distribution = loader.get_class_distribution()
    for class_name, count in sorted(distribution.items()):
        print(f"  {class_name}: {count} samples")

    # 특정 클래스 로드 테스트
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        samples = loader.load_by_class(test_class)

        if samples:
            print(f"\n=== Loaded {len(samples)} samples for '{test_class}' ===")
            for i, sample in enumerate(samples[:3]):
                print(f"  [{i}] {sample.image_path.name} - {sample.class_name}")
                print(f"      Bbox: {sample.bbox_pixel}")
                print(f"      Crop size: {sample.crop_image.shape}")
