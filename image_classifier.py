#!/usr/bin/env python3
"""
Image Classifier using Qwen3-VL
YOLO에서 crop된 이미지를 VLM으로 분류하고 카테고리별 폴더로 정리
- 동적 유사도 기반 카테고리 매핑
- 파일명: {category}_{timestamp}_{random}.ext
"""

import os
import sys
import yaml
import json
import shutil
import logging
import argparse
import string
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

# verifier 모듈 임포트 (ModelManager 재사용)
import verifier


@dataclass
class ClassificationResult:
    """이미지 분류 결과"""
    original_filename: str
    detected_category: str      # VLM이 감지한 원본 카테고리명
    mapped_category: str         # 유사도 매핑된 최종 카테고리명
    confidence: float
    response: str                # VLM 원본 응답
    similarity_score: float      # 기존 폴더와의 유사도 (새 폴더면 0.0)
    matched_folder: Optional[str]  # 매칭된 기존 폴더명 (없으면 None)
    new_filename: str            # 저장된 파일명
    timestamp: str


class CategoryMapper:
    """카테고리 유사도 기반 매핑 관리"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 전체 설정
        """
        self.config = config
        self.similarity_config = config.get('similarity', {})
        self.folder_config = config.get('folder_naming', {})

        self.method = self.similarity_config.get('method', 'difflib')
        self.threshold = self.similarity_config.get('threshold', 0.70)
        self.case_sensitive = self.similarity_config.get('case_sensitive', False)

        # 폴더명 정규화 설정
        self.normalize = self.folder_config.get('normalize', True)
        self.replace_spaces = self.folder_config.get('replace_spaces', '_')
        self.remove_special = self.folder_config.get('remove_special_chars', True)

    def normalize_category_name(self, name: str) -> str:
        """카테고리명 정규화

        Args:
            name: 원본 카테고리명

        Returns:
            정규화된 카테고리명
        """
        if not self.normalize:
            return name

        # 소문자 변환 (case_sensitive가 False일 때)
        if not self.case_sensitive:
            name = name.lower()

        # 공백 처리
        name = name.strip()
        if self.replace_spaces:
            name = name.replace(' ', self.replace_spaces)

        # 특수문자 제거
        if self.remove_special:
            allowed = string.ascii_letters + string.digits + self.replace_spaces + '-'
            name = ''.join(c for c in name if c in allowed)

        return name

    def compute_similarity(self, str1: str, str2: str) -> float:
        """두 문자열 간 유사도 계산

        Args:
            str1: 첫 번째 문자열
            str2: 두 번째 문자열

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        # 정규화
        s1 = self.normalize_category_name(str1)
        s2 = self.normalize_category_name(str2)

        if self.method == 'difflib':
            return SequenceMatcher(None, s1, s2).ratio()

        elif self.method == 'levenshtein':
            # Levenshtein distance 기반
            try:
                import Levenshtein
                distance = Levenshtein.distance(s1, s2)
                max_len = max(len(s1), len(s2))
                if max_len == 0:
                    return 1.0
                return 1.0 - (distance / max_len)
            except ImportError:
                logging.warning("python-Levenshtein not installed, falling back to difflib")
                return SequenceMatcher(None, s1, s2).ratio()

        else:
            logging.warning(f"Unknown similarity method: {self.method}, using difflib")
            return SequenceMatcher(None, s1, s2).ratio()

    def find_best_match(
        self,
        detected_name: str,
        existing_folders: List[str]
    ) -> Tuple[Optional[str], float]:
        """기존 폴더 중 가장 유사한 것 찾기

        Args:
            detected_name: VLM이 감지한 카테고리명
            existing_folders: 기존 폴더명 리스트

        Returns:
            (best_match_folder, similarity_score)
            매칭 없으면 (None, 0.0)
        """
        if not existing_folders:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for folder in existing_folders:
            score = self.compute_similarity(detected_name, folder)
            if score > best_score:
                best_score = score
                best_match = folder

        # 임계값 확인
        if best_score >= self.threshold:
            return best_match, best_score
        else:
            return None, 0.0

    def get_canonical_name(self, detected_name: str) -> str:
        """정규화된 폴더명 반환

        Args:
            detected_name: VLM이 감지한 카테고리명

        Returns:
            폴더명으로 사용할 정규화된 이름
        """
        return self.normalize_category_name(detected_name)


class ImageClassifier:
    """이미지 분류 시스템"""

    def __init__(self, config: Dict, model_manager):
        """
        Args:
            config: 전체 설정
            model_manager: verifier.ModelManager 인스턴스
        """
        self.config = config
        self.model_manager = model_manager
        self.category_mapper = CategoryMapper(config)

        # 경로 설정
        self.input_folder = Path(config['paths']['input_folder'])
        self.output_base = Path(config['paths']['output_base'])
        self.output_base.mkdir(parents=True, exist_ok=True)

        # VLM 설정
        self.vlm_config = config.get('vlm', {})
        self.prompt = self.vlm_config.get(
            'prompt',
            "What is the main object in this image? Answer with one or two words only."
        )
        self.confidence_threshold = self.vlm_config.get('confidence_threshold', 0.5)
        self.max_image_size = self.vlm_config.get('max_image_size', 1280)
        self.create_unknown = self.vlm_config.get('create_unknown_folder', True)

        # 폐색 처리 설정
        self.occlusion_config = self.vlm_config.get('occlusion_handling', {})
        self.occlusion_enabled = self.occlusion_config.get('enabled', False)
        self.occlusion_aware_prompt = self.occlusion_config.get(
            'occlusion_aware_prompt',
            self.prompt  # fallback to default
        )
        self.occlusion_threshold = self.occlusion_config.get('confidence_threshold_occluded', 0.4)

        if self.occlusion_enabled:
            logging.info("✓ Occlusion-aware classification enabled")
            logging.info(f"  - Using specialized prompt for partially hidden persons")
            logging.info(f"  - Occluded person confidence threshold: {self.occlusion_threshold}")

        # 파일명 설정
        self.filename_config = config.get('filename', {})
        self.timestamp_format = self.filename_config.get('timestamp_format', '%Y%m%d%H%M%S')
        self.random_length = self.filename_config.get('random_length', 6)

        # 처리 옵션
        self.processing_config = config.get('processing', {})
        self.copy_or_move = self.processing_config.get('copy_or_move', 'copy')
        self.skip_existing = self.processing_config.get('skip_existing', True)

        # 결과 저장
        self.results: List[ClassificationResult] = []
        self.category_stats = defaultdict(int)  # 카테고리별 이미지 수
        self.category_variants = defaultdict(set)  # 카테고리별 감지된 변형들

        # 모델 로드 확인
        if self.model_manager.model is None or self.model_manager.processor is None:
            logging.info("Loading model...")
            self.model_manager.load_model()

    def get_existing_folders(self) -> List[str]:
        """output_base 내 기존 폴더 목록 가져오기

        Returns:
            폴더명 리스트
        """
        if not self.output_base.exists():
            return []

        folders = [
            d.name for d in self.output_base.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        return folders

    def classify_image(self, image_path: Path) -> Optional[ClassificationResult]:
        """단일 이미지 분류

        Args:
            image_path: 이미지 파일 경로

        Returns:
            ClassificationResult 또는 None (실패시)
        """
        try:
            # 이미지 로드
            pil_image = Image.open(image_path).convert('RGB')

            # 이미지 리사이즈 (필요시)
            if max(pil_image.size) > self.max_image_size:
                pil_image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)

            # VLM 쿼리
            detected_category, confidence, response = self._query_vlm(pil_image)

            # 낮은 confidence 처리
            # 폐색 처리: person 감지 시 낮은 임계값 사용
            is_person = detected_category.lower() in ['person', 'people', 'human', 'pedestrian', 'man', 'woman', 'child']
            threshold_to_use = self.occlusion_threshold if (self.occlusion_enabled and is_person) else self.confidence_threshold

            if self.occlusion_enabled and is_person:
                logging.debug(f"{image_path.name}: Using occlusion threshold {threshold_to_use} for person (standard: {self.confidence_threshold})")

            if confidence < threshold_to_use:
                if self.create_unknown:
                    detected_category = "unknown"
                else:
                    logging.warning(f"Low confidence ({confidence:.2f}) for {image_path.name}, skipping")
                    return None

            # 기존 폴더와 매칭
            existing_folders = self.get_existing_folders()
            matched_folder, similarity = self.category_mapper.find_best_match(
                detected_category, existing_folders
            )

            # 최종 카테고리 결정
            if matched_folder:
                mapped_category = matched_folder
                logging.debug(
                    f"{image_path.name}: '{detected_category}' → '{mapped_category}' "
                    f"(similarity: {similarity:.2f})"
                )
            else:
                # 새 폴더 생성
                mapped_category = self.category_mapper.get_canonical_name(detected_category)
                logging.debug(f"{image_path.name}: '{detected_category}' → NEW folder '{mapped_category}'")

            # 타임스탬프
            timestamp = datetime.now().strftime(self.timestamp_format)

            # 랜덤 문자열
            random_str = ''.join(
                random.choices(string.ascii_lowercase + string.digits, k=self.random_length)
            )

            # 새 파일명 생성
            ext = image_path.suffix
            new_filename = f"{mapped_category}_{timestamp}_{random_str}{ext}"

            # 결과 생성
            result = ClassificationResult(
                original_filename=image_path.name,
                detected_category=detected_category,
                mapped_category=mapped_category,
                confidence=confidence,
                response=response,
                similarity_score=similarity,
                matched_folder=matched_folder,
                new_filename=new_filename,
                timestamp=timestamp
            )

            return result

        except Exception as e:
            logging.error(f"Failed to classify {image_path.name}: {e}")
            return None

    def save_classified_image(self, source_path: Path, result: ClassificationResult):
        """분류된 이미지 저장

        Args:
            source_path: 원본 이미지 경로
            result: 분류 결과
        """
        # 카테고리 폴더 생성
        category_folder = self.output_base / result.mapped_category
        category_folder.mkdir(parents=True, exist_ok=True)

        # 목적지 경로
        dest_path = category_folder / result.new_filename

        # 복사 또는 이동
        if self.copy_or_move == 'copy':
            shutil.copy2(source_path, dest_path)
        elif self.copy_or_move == 'move':
            shutil.move(str(source_path), str(dest_path))
        else:
            logging.warning(f"Unknown operation: {self.copy_or_move}, using copy")
            shutil.copy2(source_path, dest_path)

        logging.debug(f"Saved: {dest_path}")

    def process_folder(self):
        """입력 폴더의 모든 이미지 처리"""
        # 지원 이미지 포맷
        image_formats = self.config.get('image_formats', ['.jpg', '.jpeg', '.png', '.bmp'])
        image_formats = tuple(fmt.lower() for fmt in image_formats)

        # 이미지 파일 검색
        image_files = [
            f for f in self.input_folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_formats
        ]

        if not image_files:
            logging.error(f"No images found in {self.input_folder}")
            return

        logging.info(f"Found {len(image_files)} images to process")

        # 진행률 표시
        show_progress = self.processing_config.get('show_progress', True)
        iterator = tqdm(image_files, desc="Classifying images", disable=not show_progress)

        success_count = 0
        failed_count = 0

        for image_path in iterator:
            # 분류
            result = self.classify_image(image_path)

            if result is None:
                failed_count += 1
                continue

            # 저장
            try:
                self.save_classified_image(image_path, result)
                self.results.append(result)

                # 통계 업데이트
                self.category_stats[result.mapped_category] += 1
                self.category_variants[result.mapped_category].add(result.detected_category)

                success_count += 1

            except Exception as e:
                logging.error(f"Failed to save {image_path.name}: {e}")
                failed_count += 1

            # GPU 캐시 비우기
            if torch.cuda.is_available():
                freq = self.processing_config.get('empty_cache_frequency', 50)
                if success_count % freq == 0:
                    torch.cuda.empty_cache()

        logging.info(f"\nProcessing complete:")
        logging.info(f"  Success: {success_count}")
        logging.info(f"  Failed: {failed_count}")

        # 리포트 생성
        if self.processing_config.get('save_report', True):
            self.save_report()

    def save_report(self):
        """분류 결과 리포트 저장"""
        report_filename = self.processing_config.get('report_filename', 'classification_report.json')
        report_path = self.output_base / report_filename

        # 카테고리별 통계
        categories_info = {}
        for category, count in self.category_stats.items():
            variants = list(self.category_variants[category])
            categories_info[category] = {
                'count': count,
                'detected_variants': variants
            }

        # 리포트 생성
        report = {
            'total_images': len(self.results),
            'successful': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'input_folder': str(self.input_folder),
                'output_folder': str(self.output_base),
                'similarity_method': self.category_mapper.method,
                'similarity_threshold': self.category_mapper.threshold
            },
            'categories': categories_info,
            'results': [asdict(r) for r in self.results]
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logging.info(f"Report saved: {report_path}")

        # 요약 출력
        logging.info("\n=== Classification Summary ===")
        for category, info in sorted(categories_info.items()):
            logging.info(f"  {category}: {info['count']} images")
            if len(info['detected_variants']) > 1:
                logging.info(f"    Variants: {', '.join(info['detected_variants'])}")

    def _query_vlm(self, image: Image.Image) -> Tuple[str, float, str]:
        """VLM에게 이미지 분류 질문

        Args:
            image: PIL 이미지

        Returns:
            (detected_category, confidence, raw_response)
        """
        max_retries = self.processing_config.get('max_retries', 3)
        retry_delay = self.processing_config.get('retry_delay', 1.0)

        for attempt in range(max_retries):
            try:
                # 폐색 처리: 폐색 인식 프롬프트 사용
                prompt_to_use = self.occlusion_aware_prompt if self.occlusion_enabled else self.prompt

                # 메시지 구성
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_to_use}
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

                # 생성
                temperature = self.vlm_config.get('temperature', 0.3)
                max_new_tokens = self.vlm_config.get('max_new_tokens', 32)

                with torch.no_grad():
                    outputs = self.model_manager.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
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

                # 카테고리 추출 (응답에서 첫 1-2 단어)
                detected_category = self._extract_category(response)

                # Confidence 계산
                confidence = self._compute_confidence(outputs, inputs)

                return detected_category, confidence, response

            except Exception as e:
                logging.warning(f"VLM query failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                else:
                    logging.error(f"All retries failed")
                    return "unknown", 0.0, f"ERROR: {str(e)}"

    def _extract_category(self, response: str) -> str:
        """VLM 응답에서 카테고리명 추출

        Args:
            response: VLM 원본 응답

        Returns:
            카테고리명 (1-2 단어)
        """
        # 전처리
        response = response.strip()

        # 불필요한 접두사 제거
        prefixes = ["The main object is", "It is", "This is", "A", "An", "The"]
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # 첫 1-2 단어 추출
        words = response.split()
        if len(words) == 0:
            return "unknown"
        elif len(words) == 1:
            return words[0].lower()
        else:
            # 2단어까지 추출 (예: "safety helmet")
            category = ' '.join(words[:2]).lower()
            # 마침표, 쉼표 제거
            category = category.rstrip('.,;:!?')
            return category

    def _compute_confidence(self, outputs, inputs) -> float:
        """Confidence 계산 (verifier.py 로직 재사용)

        Args:
            outputs: 모델 출력
            inputs: 모델 입력

        Returns:
            Confidence (0.0 ~ 1.0)
        """
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


def main():
    parser = argparse.ArgumentParser(
        description='Classify cropped images using Qwen3-VL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 사용
  python image_classifier.py --input ./cropped_images

  # 설정 파일 지정
  python image_classifier.py --input ./cropped_images --config my_config.yaml

  # 특정 모델 사용
  python image_classifier.py --input ./cropped_images --model "Qwen/Qwen3-VL-4B-Instruct"
        """
    )

    parser.add_argument('--input', type=str, help='Input folder containing cropped images')
    parser.add_argument('--output', type=str, help='Output base folder (overrides config)')
    parser.add_argument('--config', type=str, default='classifier_config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, help='Model ID (overrides config)')
    parser.add_argument('--threshold', type=float, help='Similarity threshold (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        logging.info("Please create classifier_config.yaml or specify --config")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # CLI 인자로 config 오버라이드
    if args.input:
        config['paths']['input_folder'] = args.input
    if args.output:
        config['paths']['output_base'] = args.output
    if args.threshold is not None:
        config['similarity']['threshold'] = args.threshold
    if args.model:
        config['model']['selected_model'] = args.model

    # 입력 폴더 확인
    input_folder = Path(config['paths']['input_folder'])
    if not input_folder.exists():
        logging.error(f"Input folder not found: {input_folder}")
        return

    # 모델 로드
    logging.info("Loading VLM model...")
    model_manager = verifier.ModelManager(config)
    model_manager.load_model()

    # 분류 시작
    classifier = ImageClassifier(config, model_manager)

    logging.info(f"\n{'='*80}")
    logging.info(f"Image Classification Settings")
    logging.info(f"{'='*80}")
    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {classifier.output_base}")
    logging.info(f"Similarity method: {classifier.category_mapper.method}")
    logging.info(f"Similarity threshold: {classifier.category_mapper.threshold}")
    logging.info(f"Confidence threshold: {classifier.confidence_threshold}")
    logging.info(f"{'='*80}\n")

    # 처리 실행
    classifier.process_folder()

    logging.info("\n✓ Classification complete!")
    logging.info(f"✓ Results saved to: {classifier.output_base}")


if __name__ == "__main__":
    main()
