#!/usr/bin/env python3
"""
YOLO Label Verifier using Qwen3-VL
SAM3로 생성된 YOLO 라벨을 VLM으로 검증하는 시스템
- 자동 라벨 수정 기능
- 장면 분류 (실내/실외, 주간/야간)
- 개선된 디버그 시각화
"""

import os
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import sys

# ✅ CUDA_PATH 설정 (본인의 CUDA 설치 경로로 수정)
if os.environ.get("CUDA_PATH") is None:
    # CUDA 경로 자동 탐지
    possible_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    ]
    
    for cuda_path in possible_paths:
        if os.path.exists(cuda_path):
            os.environ["CUDA_PATH"] = cuda_path
            print(f"[INFO] CUDA_PATH set to: {cuda_path}")
            break
    else:
        # CUDA를 찾지 못한 경우 경고
        print("[WARNING] CUDA not found. Setting dummy path for Triton.")
        os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

# ✅ Triton 비활성화 (CUDA가 없거나 Triton이 필요 없는 경우)
os.environ["DISABLE_TORCH_TRITON"] = "1"

import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class BoundingBox:
    """YOLO 바운딩 박스"""
    class_id: int
    x_center: float  # normalized
    y_center: float  # normalized
    width: float     # normalized
    height: float    # normalized
    
    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """픽셀 좌표로 변환 (x1, y1, x2, y2)"""
        x_center_px = self.x_center * img_width
        y_center_px = self.y_center * img_height
        w_px = self.width * img_width
        h_px = self.height * img_height
        
        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)
        
        return x1, y1, x2, y2


@dataclass
class VerificationResult:
    """검증 결과"""
    image_path: str
    box_index: int
    class_id: int
    class_name: str
    is_correct: bool
    confidence: float
    response: str
    category: str  # "correct", "mislabeled", "uncertain"
    suggested_class_id: Optional[int] = None
    suggested_class_name: Optional[str] = None
    correction_confidence: float = 0.0
    detected_object_raw: Optional[str] = None  # VLM이 감지한 원본 객체명


@dataclass
class SceneInfo:
    """장면 정보"""
    location: str  # "indoor", "outdoor", "unknown"
    time: str      # "day", "night", "unknown"
    confidence: float


class ModelManager:
    """모델 다운로드 및 관리"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config['model']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.processor = None
        self.device = self._get_device()
        
    def _get_device(self):
        """디바이스 결정"""
        device_config = self.config['processing']['device']
        if device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_config
    
    def show_available_models(self) -> List[Dict]:
        """사용 가능한 모델 목록 표시"""
        models = self.config['available_models']
        print("\n" + "="*80)
        print("Available Qwen3-VL Models")
        print("="*80)
        
        for idx, model in enumerate(models, 1):
            print(f"\n[{idx}] {model['name']}")
            print(f"    Size: {model['size']}")
            print(f"    VRAM: {model['vram']}")
            print(f"    Description: {model['description']}")
            print(f"    Model ID: {model['id']}")
        
        print("\n" + "="*80)
        return models
    
    def select_model(self) -> str:
        """모델 선택 UI"""
        models = self.show_available_models()
        
        while True:
            try:
                choice = input("\nSelect model number (1-{}): ".format(len(models)))
                idx = int(choice) - 1
                
                if 0 <= idx < len(models):
                    selected = models[idx]
                    print(f"\n✓ Selected: {selected['name']}")
                    
                    # 확인
                    confirm = input("Download and use this model? (y/n): ").lower()
                    if confirm == 'y':
                        return selected['id']
                    
                else:
                    print("Invalid choice. Please try again.")
                    
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n\nCancelled by user.")
                exit(0)
    
    def load_model(self, model_id: Optional[str] = None):
        """모델 로드 (오프라인 모드 자동 감지)"""
        if model_id is None:
            model_id = self.config['model'].get('selected_model')
            if model_id is None:
                # 자동으로 다운로드된 모델 찾기
                logging.info("No model specified, searching in cache directory...")
                model_id = self._find_cached_model()
                if model_id is None:
                    # 대화형 모드에서만 선택 UI 표시
                    import sys
                    if sys.stdin.isatty():
                        model_id = self.select_model()
                        self.config['model']['selected_model'] = model_id
                    else:
                        raise RuntimeError(
                            "No model specified and no cached model found. "
                            "Please set 'selected_model' in config.yaml or run verifier.py once to download a model."
                        )
                else:
                    logging.info(f"Found cached model: {model_id}")
                    self.config['model']['selected_model'] = model_id
        
        # 오프라인 모드 감지
        is_offline = self._is_offline_mode()
        
        # 오프라인 모드에서 model_id가 모델 ID 형식(Qwen/...)이면 로컬 경로로 변환
        if is_offline and model_id.startswith("Qwen/"):
            logging.info("Offline mode detected, resolving local path...")
            local_path = self._find_cached_model()
            if local_path and not local_path.startswith("Qwen/"):
                model_id = local_path
                logging.info(f"Using local path: {model_id}")
            else:
                logging.warning(f"Could not resolve local path, using model ID: {model_id}")
        
        logging.info(f"Loading model: {model_id}")
        logging.info(f"Cache directory: {self.cache_dir}")
        if is_offline:
            logging.info("Offline mode: enabled")
        
        try:
            # 완전 오프라인 모드 환경변수 설정
            import os
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            
            # 모델 로드
            import torch
            from pathlib import Path
            
            # 로컬 경로인지 확인
            is_local_path = Path(model_id).exists() if not model_id.startswith("Qwen/") else False
            
            if is_local_path:
                # 로컬 경로: cache_dir 파라미터 제거
                logging.info("Loading from local path (offline mode)")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            else:
                # 모델 ID: cache_dir 사용
                logging.info("Loading from model ID with cache")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                    local_files_only=True,
                    force_download=False,
                    resume_download=False,
                )
            
            # 수동으로 GPU에 로드
            self.model = self.model.to(self.device)
            
            # 프로세서 로드
            if is_local_path:
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir),
                    local_files_only=True,
                    trust_remote_code=True,
                    force_download=False,
                    resume_download=False,
                )
            
            logging.info(f"✓ Model loaded successfully on {self.device}")
            
            # VRAM 사용량 확인
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def _is_offline_mode(self) -> bool:
        """오프라인 모드 감지"""
        import os
        offline_vars = [
            'HF_HUB_OFFLINE',
            'TRANSFORMERS_OFFLINE',
            'HF_DATASETS_OFFLINE'
        ]
        return any(os.environ.get(var, '0') == '1' for var in offline_vars)
    
    def _find_cached_model(self) -> Optional[str]:
        """캐시 디렉토리에서 다운로드된 모델 찾기
        
        오프라인 모드일 때는 실제 snapshot 디렉토리의 절대 경로 반환
        온라인 모드일 때는 모델 ID (예: Qwen/Qwen3-VL-8B-Instruct) 반환
        """
        try:
            is_offline = self._is_offline_mode()
            qwen_models = []
            
            # Hugging Face 캐시 형식: models/models--Qwen--Qwen3-VL-*
            hf_models = list(self.cache_dir.glob("models--Qwen--Qwen3-VL-*"))
            if hf_models:
                for model_dir in hf_models:
                    model_name = model_dir.name.replace("models--Qwen--", "")
                    
                    if is_offline:
                        # 오프라인: snapshot 경로 찾기
                        snapshots_dir = model_dir / "snapshots"
                        if snapshots_dir.exists():
                            snapshot_dirs = list(snapshots_dir.iterdir())
                            if snapshot_dirs:
                                # 가장 최근 snapshot 사용 (첫 번째)
                                snapshot_path = snapshot_dirs[0]
                                qwen_models.append((model_name, str(snapshot_path.absolute())))
                    else:
                        # 온라인: 모델 ID 사용
                        qwen_models.append((model_name, f"Qwen/{model_name}"))
            
            # 직접 다운로드 형식: models/Qwen/Qwen3-VL-*
            elif (self.cache_dir / "Qwen").exists():
                for model_dir in (self.cache_dir / "Qwen").glob("Qwen3-VL-*"):
                    model_name = model_dir.name
                    
                    if is_offline:
                        # 오프라인: 절대 경로 사용
                        qwen_models.append((model_name, str(model_dir.absolute())))
                    else:
                        # 온라인: 모델 ID 사용
                        qwen_models.append((model_name, f"Qwen/{model_name}"))
            
            if qwen_models:
                # 권장 순서: 8B > 4B > 2B > 기타
                for preferred in ["Qwen3-VL-8B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-2B-Instruct"]:
                    for model_name, model_path in qwen_models:
                        if preferred in model_name:
                            return model_path
                # 첫 번째 모델 사용
                return qwen_models[0][1]
            
            return None
            
        except Exception as e:
            logging.warning(f"Error searching for cached models: {e}")
            return None


class YOLOParser:
    """YOLO 라벨 파일 파싱"""
    
    @staticmethod
    def parse_label_file(label_path: Path) -> List[BoundingBox]:
        """YOLO txt 파일 파싱"""
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
                        
                        boxes.append(BoundingBox(
                            class_id=class_id,
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height
                        ))
        
        except Exception as e:
            logging.error(f"Error parsing label file {label_path}: {e}")
        
        return boxes


class LabelVerifier:
    """라벨 검증 엔진"""
    
    def __init__(self, model_manager: ModelManager, config: Dict):
        self.model_manager = model_manager
        self.config = config
        
        # 모델이 아직 로드되지 않았으면 로드
        if self.model_manager.model is None or self.model_manager.processor is None:
            logging.info("Loading model...")
            self.model_manager.load_model()
        
        # Classes 로드 (리스트 형식 지원)
        raw_classes = config['classes']
        self.class_map = {}  # class_id -> primary_name
        self.class_aliases = {}  # alias (lowercase) -> (class_id, primary_name)
        
        for class_id, names in raw_classes.items():
            class_id = int(class_id)
            
            # 리스트 또는 단일 문자열 지원
            if isinstance(names, list):
                name_list = names
            else:
                name_list = [names]
            
            # 첫 번째 이름이 대표 이름
            primary_name = name_list[0]
            self.class_map[class_id] = primary_name
            
            # 모든 이름(동의어)을 alias로 등록
            for name in name_list:
                self.class_aliases[name.lower()] = (class_id, primary_name)
        
        self.verification_config = config['verification']
        self.processing_config = config['processing']

        # 폐색 처리 설정 로드
        self.occlusion_config = self.verification_config.get('occlusion_handling', {})
        self.occlusion_enabled = self.occlusion_config.get('enabled', False)
        self.occlusion_threshold = self.occlusion_config.get('confidence_threshold_occluded', 0.5)
        self.occlusion_classes = set(name.lower() for name in self.occlusion_config.get('classes_with_occlusion', []))

        if self.occlusion_enabled:
            logging.info(f"✓ Occlusion handling enabled for classes: {self.occlusion_classes}")
            logging.info(f"  - Standard confidence threshold: {self.verification_config['confidence_threshold']}")
            logging.info(f"  - Occluded objects threshold: {self.occlusion_threshold}")

        # 통계
        self.stats = {
            'total_images': 0,
            'total_boxes': 0,
            'correct': 0,
            'mislabeled': 0,
            'uncertain': 0,
            'failed': 0,
            'per_class': {},
            'per_scene': {
                'indoor': {'day': 0, 'night': 0, 'unknown': 0},
                'outdoor': {'day': 0, 'night': 0, 'unknown': 0},
                'unknown': {'day': 0, 'night': 0, 'unknown': 0}
            }
        }
        
        # 클래스별 통계 초기화
        for class_id, class_name in self.class_map.items():
            self.stats['per_class'][class_name] = {
                'correct': 0,
                'mislabeled': 0,
                'uncertain': 0
            }

        # 최적화된 프롬프트 로드
        self.optimized_prompts = {}
        prompt_opt_config = config.get('prompt_optimization', {})
        use_optimized = prompt_opt_config.get('production', {}).get('use_optimized_prompts', False)

        if use_optimized:
            self._load_optimized_prompts()

    def _load_optimized_prompts(self):
        """최적화된 프롬프트 템플릿 로드"""
        try:
            templates_dir = Path(self.config.get('prompt_optimization', {}).get('templates', {}).get('dir', 'prompt_discovery/templates'))
            current_template = templates_dir / 'current.yaml'

            if not current_template.exists():
                logging.warning(f"Optimized prompts not found: {current_template}")
                logging.warning("Using default prompts. Run prompt optimization first.")
                return

            with open(current_template, 'r', encoding='utf-8') as f:
                template = yaml.safe_load(f)

            # 'prompts' 키가 있으면 그 안에서 로드, 없으면 루트에서 로드
            prompts_dict = template.get('prompts', template) if isinstance(template, dict) else {}

            for class_name, prompt_data in prompts_dict.items():
                # 코멘트나 메타데이터 무시
                if class_name.startswith('#'):
                    continue

                if isinstance(prompt_data, dict) and 'prompt' in prompt_data:
                    self.optimized_prompts[class_name] = prompt_data['prompt']
                    logging.info(f"Loaded optimized prompt for '{class_name}' (acc: {prompt_data.get('accuracy', 'N/A')})")
                elif isinstance(prompt_data, str):
                    self.optimized_prompts[class_name] = prompt_data
                    logging.debug(f"Loaded prompt for '{class_name}'")

            logging.info(f"✓ Loaded {len(self.optimized_prompts)} optimized prompts")

        except Exception as e:
            logging.error(f"Failed to load optimized prompts: {e}")

    def _build_prompt(self, class_name: str) -> str:
        """프롬프트 생성 (최적화된 프롬프트 우선)"""
        # 1. 최적화된 프롬프트
        if class_name in self.optimized_prompts:
            return self.optimized_prompts[class_name]

        # 2. Fallback 프롬프트
        fallback = self.config.get('prompt_optimization', {}).get('production', {}).get('fallback_prompt', None)
        if fallback:
            return fallback.format(class_name=class_name)

        # 3. 기본 프롬프트
        return f"Is this object a {class_name}? Answer only 'Yes' or 'No'."

    def crop_box(self, image: np.ndarray, box: BoundingBox, padding: int = 10) -> np.ndarray:
        """박스 영역 크롭"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box.to_pixel_coords(w, h)
        
        # 패딩 추가
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # 최소 크기 확인
        min_size = self.verification_config['crop_min_size']
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            # 크기가 너무 작으면 중심 기준으로 최소 크기 확보
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            half_size = min_size // 2
            
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(w, center_x + half_size)
            y2 = min(h, center_y + half_size)
        
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def ask_model(self, image: Image.Image, class_name: str, retries: int = 0) -> Tuple[bool, float, str]:
        """모델에게 질문하고 실제 confidence 계산 (Logit 기반)"""
        prompt = self._build_prompt(class_name)

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
                    output_scores=True,           # ← 실제 확률 계산용
                    return_dict_in_generate=True  # ← 딕셔너리 반환
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
            
            # 실제 confidence 계산 (Logit 기반)
            real_confidence = self._compute_real_confidence(outputs, inputs)
            
            # 디버깅 로그
            logging.debug(f"Response: '{response}' | is_correct: {is_correct} | real_confidence: {real_confidence:.4f}")
            
            return is_correct, real_confidence, response
            
        except Exception as e:
            logging.error(f"Model query failed: {e}")
            
            # 재시도
            max_retries = self.processing_config['max_retries']
            if retries < max_retries:
                import time
                time.sleep(self.processing_config['retry_delay'])
                return self.ask_model(image, class_name, retries + 1)
            
            return False, 0.0, f"ERROR: {str(e)}"
    
    def _parse_yes_no(self, response: str) -> Optional[bool]:
        """응답에서 Yes/No만 파싱 (confidence는 logit으로 계산)"""
        response_lower = response.lower()
        
        # Yes/No 판단
        has_yes = 'yes' in response_lower
        has_no = 'no' in response_lower
        
        if has_yes and not has_no:
            return True
        elif has_no and not has_yes:
            return False
        else:
            # 애매한 경우
            return None
    
    def _compute_real_confidence(self, outputs, inputs) -> float:
        """실제 토큰 확률 기반 confidence 계산 (Logit 기반)
        
        Args:
            outputs: model.generate()의 반환값 (output_scores=True, return_dict_in_generate=True)
            inputs: 모델 입력
            
        Returns:
            float: 평균 토큰 확률 (0.0 ~ 1.0)
        """
        try:
            # Transition scores 계산 (각 토큰의 log 확률)
            transition_scores = self.model_manager.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            
            # Log 확률 → 실제 확률
            # transition_scores[0]: 첫 번째 배치의 모든 토큰
            token_probs = torch.exp(transition_scores[0])
            
            # 평균 확률 계산
            real_confidence = token_probs.mean().item()
            
            # 최소값도 확인 (가장 불확실했던 토큰)
            min_prob = token_probs.min().item()
            
            # 디버깅 로그
            logging.debug(f"Token probs: mean={real_confidence:.4f}, min={min_prob:.4f}, all={token_probs.tolist()}")
            
            # 0~1 범위 보장
            real_confidence = max(0.0, min(1.0, real_confidence))
            
            return real_confidence
            
        except Exception as e:
            logging.warning(f"Failed to compute real confidence: {e}")
            # Fallback: 중간값 반환
            return 0.5
    
    def ask_what_is_this(self, image: Image.Image, retries: int = 0) -> Tuple[Optional[str], float]:
        """잘못된 라벨의 경우 실제 객체가 무엇인지 질문 (Logit 기반 confidence)"""
        prompt = "What is this object? Answer with a single word or short phrase describing the main object."
        
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
                    max_new_tokens=64,
                    temperature=0.3,
                    output_scores=True,           # ← 실제 확률 계산용
                    return_dict_in_generate=True  # ← 딕셔너리 반환
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
            
            # 응답에서 객체명 추출
            detected_name = self._extract_object_name(response)
            
            # 실제 confidence 계산 (Logit 기반)
            real_confidence = self._compute_real_confidence(outputs, inputs)
            
            # 디버깅 로그
            logging.debug(f"Detected: '{detected_name}' | real_confidence: {real_confidence:.4f}")
            
            return detected_name, real_confidence
            
        except Exception as e:
            logging.error(f"Alternative query failed: {e}")
            
            # 재시도
            max_retries = self.processing_config['max_retries']
            if retries < max_retries:
                import time
                time.sleep(self.processing_config['retry_delay'])
                return self.ask_what_is_this(image, retries + 1)
            
            return None, 0.0

    def _extract_object_name(self, response: str) -> Optional[str]:
        """응답에서 객체 이름 추출"""
        # 일반적인 패턴 제거
        response = response.lower().strip()
        
        # "this is a/an", "it is a/an", "the object is" 등 제거
        remove_patterns = [
            "this is a ",
            "this is an ",
            "it is a ",
            "it is an ",
            "the object is a ",
            "the object is an ",
            "i see a ",
            "i see an ",
            "this appears to be a ",
            "this appears to be an ",
        ]
        
        for pattern in remove_patterns:
            if response.startswith(pattern):
                response = response[len(pattern):]
                break
        
        # 마침표, 쉼표 제거
        response = response.rstrip('.,!?')
        
        # 첫 단어만 추출 (보통 객체명)
        words = response.split()
        if words:
            return words[0]
        
        return None

    def match_to_known_class(self, detected_name: str) -> Tuple[Optional[int], Optional[str], float]:
        """감지된 이름을 config의 클래스와 매칭 (동의어 지원)"""
        if not detected_name:
            return None, None, 0.0
        
        from difflib import SequenceMatcher
        
        detected_lower = detected_name.lower().strip()
        
        # 1. 동의어 딕셔너리에서 정확한 매칭
        if detected_lower in self.class_aliases:
            class_id, primary_name = self.class_aliases[detected_lower]
            logging.info(f"Alias match: '{detected_name}' -> '{primary_name}' (id={class_id})")
            return class_id, primary_name, 1.0
        
        # 2. 유사도 매칭 (모든 동의어에 대해)
        auto_config = self.config.get('auto_correction', {})
        similarity_threshold = auto_config.get('similarity_threshold', 0.8)
        
        best_match = None
        best_score = 0.0
        
        for alias, (class_id, primary_name) in self.class_aliases.items():
            ratio = SequenceMatcher(None, detected_lower, alias).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = (class_id, primary_name)
        
        if best_score >= similarity_threshold:
            logging.info(f"Similarity match: '{detected_name}' -> '{best_match[1]}' (id={best_match[0]}, score={best_score:.2f})")
            return best_match[0], best_match[1], best_score
        
        # 3. Fallback 클래스
        fallback_name = auto_config.get('fallback_class', 'unknown')
        
        # fallback 이름의 alias 검색
        if fallback_name.lower() in self.class_aliases:
            class_id, primary_name = self.class_aliases[fallback_name.lower()]
            logging.warning(f"No match for '{detected_name}', using fallback: '{primary_name}' (id={class_id})")
            return class_id, primary_name, 0.5
        
        # Fallback 클래스도 없으면
        logging.error(f"No match and no fallback class for '{detected_name}'")
        return None, None, 0.0
    
    def classify_scene(self, image: Image.Image, retries: int = 0) -> SceneInfo:
        """장면 분류: 실내/실외, 주간/야간"""
        prompt = """Analyze this image and answer two questions:
1. Is this INDOOR or OUTDOOR?
2. Is this DAY or NIGHT?
Answer in format: [INDOOR/OUTDOOR] [DAY/NIGHT]"""
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = self.model_manager.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.model_manager.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.model_manager.device)
            
            with torch.no_grad():
                generated_ids = self.model_manager.model.generate(
                    **inputs, max_new_tokens=64, temperature=0.3
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.model_manager.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 응답 파싱
            return self._parse_scene_response(response)
            
        except Exception as e:
            logging.error(f"Scene classification failed: {e}")
            
            max_retries = self.processing_config['max_retries']
            if retries < max_retries:
                import time
                time.sleep(self.processing_config['retry_delay'])
                return self.classify_scene(image, retries + 1)
            
            return SceneInfo("unknown", "unknown", 0.0)

    def _parse_scene_response(self, response: str) -> SceneInfo:
        """장면 분류 응답 파싱"""
        response_lower = response.lower()
        
        # Location
        if "indoor" in response_lower:
            location = "indoor"
        elif "outdoor" in response_lower:
            location = "outdoor"
        else:
            location = "unknown"
        
        # Time
        if "day" in response_lower or "daytime" in response_lower:
            time = "day"
        elif "night" in response_lower or "nighttime" in response_lower:
            time = "night"
        else:
            time = "unknown"
        
        # Confidence
        confidence = 0.8
        strong_words = ['clearly', 'definitely', 'obviously']
        if any(word in response_lower for word in strong_words):
            confidence = 0.95
        
        return SceneInfo(location, time, confidence)
    
    def verify_single_box(self, image_path: Path, box: BoundingBox, box_idx: int) -> VerificationResult:
        """단일 박스 검증"""
        try:
            # numpy 배열로 읽고 디코딩
            with open(image_path, 'rb') as f:
                image_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                logging.error(f"Failed to decode image: {image_path}")
                return None
                    
        except Exception as e:
            logging.error(f"Failed to load image: {image_path}, Error: {e}")
            return None
        
        # RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 크롭
        cropped = self.crop_box(
            image_rgb,
            box,
            padding=self.verification_config['crop_padding']
        )
        
        # PIL 이미지로 변환
        pil_image = Image.fromarray(cropped)
        
        # 크기 조정
        max_size = self.verification_config['max_image_size']
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 클래스 이름
        class_name = self.class_map[box.class_id]
        
        # 모델에게 질문
        is_correct, confidence, response = self.ask_model(pil_image, class_name)

        # 카테고리 결정
        # 폐색 처리: person 클래스는 낮은 신뢰도 임계값 사용
        is_occlusion_class = self.occlusion_enabled and class_name.lower() in self.occlusion_classes

        if is_occlusion_class:
            threshold = self.occlusion_threshold
            logging.debug(f"  Using occlusion threshold {threshold} for {class_name} (standard: {self.verification_config['confidence_threshold']})")
        else:
            threshold = self.verification_config['confidence_threshold']

        if is_correct is None:
            category = "uncertain"
        elif confidence < threshold:
            category = "uncertain"
        elif is_correct:
            category = "correct"
        else:
            category = "mislabeled"
        
        # 결과 생성
        result = VerificationResult(
            image_path=str(image_path),
            box_index=box_idx,
            class_id=box.class_id,
            class_name=class_name,
            is_correct=is_correct if is_correct is not None else False,
            confidence=confidence,
            response=response,
            category=category
        )
        
        # mislabeled면 추가 질문
        if category == "mislabeled":
            auto_config = self.config.get('auto_correction', {})
            if auto_config.get('enabled', False) and auto_config.get('ask_alternative', True):
                detected_name, corr_conf = self.ask_what_is_this(pil_image)
                
                if detected_name:
                    # VLM이 감지한 원본 객체명 저장
                    result.detected_object_raw = detected_name
                    
                    suggested_id, suggested_name, match_conf = self.match_to_known_class(detected_name)
                    
                    if suggested_id is not None:
                        result.suggested_class_id = suggested_id
                        result.suggested_class_name = suggested_name
                        result.correction_confidence = corr_conf * match_conf
                        
                        logging.info(f"  Box {box_idx}: {class_name} -> {suggested_name} (confidence: {result.correction_confidence:.2f})")
        
        return result
    
    def verify_image(self, image_path: Path, label_path: Path) -> Tuple[str, List[VerificationResult]]:
        """이미지의 모든 박스 검증"""
        # 라벨 파싱
        boxes = YOLOParser.parse_label_file(label_path)
        
        if not boxes:
            logging.warning(f"No boxes found in {label_path}")
            return "correct", []
        
        results = []
        
        # 모드에 따라 처리
        mode = self.verification_config['mode']
        
        if mode == "single":
            # 박스 하나씩 검증
            for idx, box in enumerate(boxes):
                result = self.verify_single_box(image_path, box, idx)
                if result:
                    results.append(result)
        
        elif mode == "batch":
            # 배치 처리 (TODO: 추후 구현)
            # 현재는 single과 동일하게 처리
            for idx, box in enumerate(boxes):
                result = self.verify_single_box(image_path, box, idx)
                if result:
                    results.append(result)
        
        # 전체 이미지 카테고리 결정
        categories = [r.category for r in results]
        
        if not categories:
            overall_category = "correct"
        elif "mislabeled" in categories:
            overall_category = "mislabeled"
        elif "uncertain" in categories and self.verification_config['create_uncertain_folder']:
            overall_category = "uncertain"
        else:
            overall_category = "correct"
        
        # 통계 업데이트
        self.stats['total_boxes'] += len(results)
        for result in results:
            self.stats[result.category] += 1
            self.stats['per_class'][result.class_name][result.category] += 1
        
        return overall_category, results


class ResultManager:
    """결과 관리 및 저장"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_base = Path(config['paths']['output_base'])
        
        # 출력 폴더 생성 (category/location/time 구조)
        self.categories = ['correct', 'mislabeled', 'uncertain']
        self.locations = ['indoor', 'outdoor', 'unknown']
        self.times = ['day', 'night', 'unknown']
        
        # 모든 조합의 폴더 생성
        for category in self.categories:
            for location in self.locations:
                for time in self.times:
                    folder = self.output_base / category / location / time
                    (folder / 'JPEGImages').mkdir(parents=True, exist_ok=True)
                    (folder / 'labels').mkdir(parents=True, exist_ok=True)
        
        if config.get('debug', {}).get('save_visualization', False):
            self.debug_folder = self.output_base / 'debug_images'
            self.debug_folder.mkdir(parents=True, exist_ok=True)
    
    def copy_files(self, image_path: Path, label_path: Path, 
                   category: str, scene_info: SceneInfo = None):
        """파일 복사 - 간단한 구조 (category만)"""
        target_folder = self.output_base / category
        
        # 이미지 복사
        target_image_folder = target_folder / 'JPEGImages'
        target_image_folder.mkdir(parents=True, exist_ok=True)
        target_image = target_image_folder / image_path.name
        
        if self.config['output']['copy_files']:
            shutil.copy2(image_path, target_image)
        else:
            if not target_image.exists():
                target_image.symlink_to(image_path.resolve())
        
        # 라벨 복사
        target_label_folder = target_folder / 'labels'
        target_label_folder.mkdir(parents=True, exist_ok=True)
        target_label = target_label_folder / label_path.name
        
        if self.config['output']['copy_files']:
            shutil.copy2(label_path, target_label)
        else:
            if not target_label.exists():
                target_label.symlink_to(label_path.resolve())
    
    def save_corrected_label(
        self, 
        label_path: Path, 
        boxes: List[BoundingBox], 
        results: List[VerificationResult],
        category: str,
        scene_info: SceneInfo = None
    ):
        """수정된 라벨 저장 - 간단한 구조 (category만)"""
        auto_config = self.config.get('auto_correction', {})
        
        if not auto_config.get('enabled', False):
            return
        
        target_folder = self.output_base / category
        
        # 원본 백업
        if auto_config.get('save_original', True):
            original_folder = target_folder / 'labels_original'
            original_folder.mkdir(parents=True, exist_ok=True)
            
            original_backup = original_folder / label_path.name
            if self.config['output']['copy_files']:
                shutil.copy2(label_path, original_backup)
        
        # 수정이 필요한지 확인
        needs_correction = any(r.suggested_class_id is not None for r in results)
        
        if not needs_correction:
            return
        
        # 수정된 라벨 생성
        corrected_boxes = []
        for box, result in zip(boxes, results):
            new_box = BoundingBox(
                class_id=result.suggested_class_id if result.suggested_class_id is not None else box.class_id,
                x_center=box.x_center,
                y_center=box.y_center,
                width=box.width,
                height=box.height
            )
            corrected_boxes.append(new_box)
        
        # 저장
        label_folder = target_folder / 'labels'
        label_folder.mkdir(parents=True, exist_ok=True)
        corrected_label_path = label_folder / label_path.name
        
        with open(corrected_label_path, 'w') as f:
            for box in corrected_boxes:
                line = f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n"
                f.write(line)
        
        logging.info(f"  Corrected label saved: {corrected_label_path.name}")
    
    def save_report(self, stats: Dict, results_log: List[Dict], start_time: datetime):
        """검증 리포트 저장"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = {
            'timestamp': end_time.isoformat(),
            'duration_seconds': duration,
            'statistics': stats,
            'config': self.config,
            'detailed_results': results_log
        }
        
        report_path = self.output_base / 'verification_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Report saved to: {report_path}")
        
        # 텍스트 요약도 저장
        self._save_summary(stats, duration)
    
    def draw_verification_results(
        self, 
        image: np.ndarray, 
        boxes: List[BoundingBox], 
        results: List[VerificationResult],
        class_map: Dict[int, str],
        scene_info: SceneInfo = None
    ) -> np.ndarray:
        """검증 결과를 이미지에 시각화 - 우측 패널 방식"""
        h, w = image.shape[:2]
        
        # 색상 정의
        colors = {
            'correct': (0, 255, 0),      # 초록
            'mislabeled': (0, 0, 255),   # 빨강
            'uncertain': (0, 255, 255)   # 노랑
        }
        
        # 1. 패널 너비 계산
        panel_width = 420
        
        # 2. 캔버스 확장 (원본 이미지 + 우측 패널)
        canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        canvas[:, :w] = image  # 원본 이미지 복사
        
        # 3. 패널 영역 (어두운 회색 배경)
        canvas[:, w:] = (40, 40, 40)
        
        # 4. 원본 이미지에 박스 그리기
        for idx, (box, result) in enumerate(zip(boxes, results)):
            x1, y1, x2, y2 = box.to_pixel_coords(w, h)
            color = colors.get(result.category, (255, 255, 255))
            
            # 박스 그리기
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            
            # 박스 번호
            cv2.circle(canvas, (x1 + 15, y1 + 15), 12, color, -1)
            cv2.putText(canvas, str(idx), (x1 + 10, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 5. 우측 패널에 정보 그리기
        panel_x = w + 15  # 패널 시작 x 좌표
        y_offset = 25
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        line_height = 20
        
        # Scene 정보
        if scene_info:
            # 구분선
            line_end_x = w + panel_width - 15
            cv2.line(canvas, (panel_x, y_offset - 5), (line_end_x, y_offset - 5),
                    (100, 100, 100), 2)
            y_offset += 10
            
            scene_text = f"Scene: {scene_info.location.upper()}"
            cv2.putText(canvas, scene_text, (panel_x, y_offset),
                       font, 0.45, (255, 255, 255), thickness + 1)
            y_offset += line_height
            
            time_text = f"Time: {scene_info.time.upper()}"
            cv2.putText(canvas, time_text, (panel_x, y_offset),
                       font, 0.45, (255, 255, 255), thickness + 1)
            y_offset += line_height
            
            conf_text = f"Confidence: {scene_info.confidence:.2f}"
            cv2.putText(canvas, conf_text, (panel_x, y_offset),
                       font, 0.38, (200, 200, 200), thickness)
            y_offset += line_height
            
            # 구분선
            cv2.line(canvas, (panel_x, y_offset + 5), (line_end_x, y_offset + 5),
                    (100, 100, 100), 2)
            y_offset += 25
        
        # 각 박스 정보
        for idx, result in enumerate(results):
            # 화면 범위 체크
            if y_offset > h - 30:
                # 더 이상 그릴 공간 없음
                remaining = len(results) - idx
                cv2.putText(canvas, f"... +{remaining} more boxes", 
                           (panel_x, y_offset),
                           font, 0.35, (150, 150, 150), 1)
                break
            
            # 박스 번호와 클래스
            header = f"[Box {idx}] {result.class_name}"
            cv2.putText(canvas, header, (panel_x, y_offset),
                       font, 0.48, (255, 255, 255), thickness + 1)
            y_offset += line_height + 2
            
            # 카테고리
            color = colors.get(result.category, (255, 255, 255))
            category_text = f"{result.category.upper()} ({result.confidence:.2f})"
            cv2.putText(canvas, category_text, (panel_x + 5, y_offset),
                       font, 0.42, color, thickness)
            y_offset += line_height
            
            # mislabeled/uncertain이면 응답 표시
            if result.category in ['mislabeled', 'uncertain']:
                # "Response:" 레이블
                cv2.putText(canvas, "Response:", (panel_x + 5, y_offset),
                           font, 0.38, (150, 150, 150), 1)
                y_offset += 18
                
                # 응답을 여러 줄로 분할 (35자마다)
                response_text = result.response
                if len(response_text) > 100:
                    response_text = response_text[:100] + "..."
                
                words = response_text.split()
                current_line = ""
                
                for word in words:
                    test_line = current_line + word + " "
                    if len(test_line) > 35:
                        if current_line:
                            cv2.putText(canvas, "  " + current_line.strip(), 
                                       (panel_x + 5, y_offset),
                                       font, 0.35, (180, 180, 180), 1)
                            y_offset += 17
                        current_line = word + " "
                    else:
                        current_line = test_line
                
                if current_line:
                    cv2.putText(canvas, "  " + current_line.strip(), 
                               (panel_x + 5, y_offset),
                               font, 0.35, (180, 180, 180), 1)
                    y_offset += 17
            
            # 수정 제안 (VLM 원본 답변 포함)
            if result.suggested_class_name:
                # "Detected:" 레이블
                cv2.putText(canvas, "Detected:", (panel_x + 5, y_offset),
                           font, 0.38, (150, 150, 150), 1)
                y_offset += 18
                
                # VLM 원본 답변 표시
                if result.detected_object_raw:
                    # "ship" -> unknown 형식
                    if result.detected_object_raw.lower() != result.suggested_class_name.lower():
                        # 원본과 매핑 결과가 다른 경우 (fallback 사용)
                        detect_text = f'  "{result.detected_object_raw}" -> {result.suggested_class_name}'
                    else:
                        # 원본과 매핑 결과가 같은 경우
                        detect_text = f'  "{result.detected_object_raw}"'
                else:
                    # 원본 정보 없으면 매핑 결과만
                    detect_text = f"  -> {result.suggested_class_name}"
                
                cv2.putText(canvas, detect_text, (panel_x + 5, y_offset),
                           font, 0.40, (255, 165, 0), thickness)
                y_offset += line_height
                
                conf_text = f"     (confidence: {result.correction_confidence:.2f})"
                cv2.putText(canvas, conf_text, (panel_x + 5, y_offset),
                           font, 0.35, (200, 140, 0), 1)
                y_offset += line_height - 2
            
            # 구분선
            line_end_x = w + panel_width - 15
            cv2.line(canvas, (panel_x, y_offset + 3), (line_end_x, y_offset + 3),
                    (80, 80, 80), 1)
            y_offset += 15
        
        return canvas
    
    def save_debug_visualization(
        self, 
        image_path: Path, 
        image: np.ndarray,
        boxes: List[BoundingBox],
        results: List[VerificationResult],
        class_map: Dict[int, str],
        scene_info: SceneInfo = None
    ):
        """디버그 시각화 이미지 저장"""
        if not self.config.get('debug', {}).get('save_visualization', False):
            return
        
        # 시각화 그리기
        vis_image = self.draw_verification_results(image, boxes, results, class_map, scene_info)
        
        # 저장
        output_path = self.debug_folder / f"{image_path.stem}_verified.jpg"
        
        # 한글 경로 처리
        success, encoded_img = cv2.imencode('.jpg', vis_image)
        if success:
            with open(output_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            
    def _save_summary(self, stats: Dict, duration: float):
        """텍스트 요약 저장"""
        summary_path = self.output_base / 'summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("YOLO Label Verification Summary\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Images: {stats['total_images']}\n")
            f.write(f"Total Boxes: {stats['total_boxes']}\n")
            f.write(f"Duration: {duration:.2f} seconds\n\n")
            
            f.write("Overall Results:\n")
            f.write(f"  Correct: {stats['correct']} ({stats['correct']/max(stats['total_boxes'],1)*100:.1f}%)\n")
            f.write(f"  Mislabeled: {stats['mislabeled']} ({stats['mislabeled']/max(stats['total_boxes'],1)*100:.1f}%)\n")
            f.write(f"  Uncertain: {stats['uncertain']} ({stats['uncertain']/max(stats['total_boxes'],1)*100:.1f}%)\n")
            f.write(f"  Failed: {stats['failed']}\n\n")
            
            f.write("Scene Distribution:\n")
            for location, times in stats['per_scene'].items():
                for time, count in times.items():
                    if count > 0:
                        f.write(f"  {location}/{time}: {count}\n")
            f.write("\n")
            
            f.write("Per-Class Results:\n")
            for class_name, class_stats in stats['per_class'].items():
                total = sum(class_stats.values())
                if total > 0:
                    f.write(f"\n  {class_name}:\n")
                    f.write(f"    Correct: {class_stats['correct']} ({class_stats['correct']/total*100:.1f}%)\n")
                    f.write(f"    Mislabeled: {class_stats['mislabeled']} ({class_stats['mislabeled']/total*100:.1f}%)\n")
                    f.write(f"    Uncertain: {class_stats['uncertain']} ({class_stats['uncertain']/total*100:.1f}%)\n")


def main():
    """메인 실행 함수"""
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 설정 로드
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logging.info("="*80)
    logging.info("YOLO Label Verifier using Qwen3-VL")
    logging.info("="*80)
    
    # 경로 확인
    input_images = Path(config['paths']['input_images'])
    input_labels = Path(config['paths']['input_labels'])
    
    if not input_images.exists():
        logging.error(f"Input images folder not found: {input_images}")
        return
    
    if not input_labels.exists():
        logging.error(f"Input labels folder not found: {input_labels}")
        return
    
    # 모델 관리자 초기화
    logging.info("\n[1/4] Initializing model manager...")
    model_manager = ModelManager(config)
    
    # 모델 로드
    logging.info("\n[2/4] Loading model...")
    model_manager.load_model()
    
    # 검증기 초기화
    logging.info("\n[3/4] Initializing verifier...")
    verifier = LabelVerifier(model_manager, config)
    result_manager = ResultManager(config)
    
    # 이미지 목록 가져오기
    image_files = sorted(input_images.glob('*.[jp][pn][g]'))
    logging.info(f"Found {len(image_files)} images")
    
    if not image_files:
        logging.error("No images found!")
        return
    
    # 검증 시작
    logging.info("\n[4/4] Starting verification...")
    start_time = datetime.now()
    
    results_log = []
    processed_count = 0
    
    # 진행률 표시
    show_progress = config['processing']['show_progress']
    iterator = tqdm(image_files, desc="Verifying") if show_progress else image_files
    
    for image_path in iterator:
        # 라벨 파일 찾기
        label_path = input_labels / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            logging.warning(f"Label not found for {image_path.name}")
            continue
        
        try:
            # 이미지 로드 (한글 처리)
            with open(image_path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                logging.error(f"Failed to decode image: {image_path}")
                continue
            
            # Scene 분류
            scene_config = config.get('scene_classification', {})
            if scene_config.get('enabled', True):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                scene_info = verifier.classify_scene(pil_img)
                logging.info(f"Scene: {scene_info.location}/{scene_info.time} ({scene_info.confidence:.2f})")
                
                # 장면 통계 업데이트
                verifier.stats['per_scene'][scene_info.location][scene_info.time] += 1
            else:
                scene_info = SceneInfo("unknown", "unknown", 0.0)
            
            # 검증
            category, box_results = verifier.verify_image(image_path, label_path)

            if box_results:
                boxes = YOLOParser.parse_label_file(label_path)
                result_manager.save_debug_visualization(
                    image_path, img, boxes, box_results, 
                    config['classes'], scene_info)
            
            # 파일 복사 (scene 정보 포함)
            result_manager.copy_files(image_path, label_path, category, scene_info)

            # 수정된 라벨 저장
            if box_results:
                result_manager.save_corrected_label(
                    label_path, boxes, box_results, category, scene_info)
            
            # 로그 저장
            for result in box_results:
                results_log.append({
                    'image': result.image_path,
                    'box_index': result.box_index,
                    'class_id': result.class_id,
                    'class_name': result.class_name,
                    'is_correct': result.is_correct,
                    'confidence': result.confidence,
                    'category': result.category,
                    'response': result.response,
                    'suggested_class_id': result.suggested_class_id,
                    'suggested_class_name': result.suggested_class_name,
                    'correction_confidence': result.correction_confidence,
                    'detected_object_raw': result.detected_object_raw,
                    'scene_location': scene_info.location,
                    'scene_time': scene_info.time
                })
            
            verifier.stats['total_images'] += 1
            processed_count += 1
            
            # 주기적으로 캐시 비우기
            if torch.cuda.is_available():
                cache_freq = config['processing']['empty_cache_frequency']
                if processed_count % cache_freq == 0:
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logging.error(f"Error processing {image_path.name}: {e}")
            verifier.stats['failed'] += 1
            continue
    
    # 리포트 생성
    logging.info("\nGenerating report...")
    result_manager.save_report(verifier.stats, results_log, start_time)
    
    # 최종 요약 출력
    logging.info("\n" + "="*80)
    logging.info("Verification Complete!")
    logging.info("="*80)
    logging.info(f"Total Images: {verifier.stats['total_images']}")
    logging.info(f"Total Boxes: {verifier.stats['total_boxes']}")
    logging.info(f"Correct: {verifier.stats['correct']}")
    logging.info(f"Mislabeled: {verifier.stats['mislabeled']}")
    logging.info(f"Uncertain: {verifier.stats['uncertain']}")
    logging.info(f"Failed: {verifier.stats['failed']}")
    logging.info(f"\nResults saved to: {config['paths']['output_base']}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
