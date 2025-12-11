#!/bin/bash

################################################################################
# YOLO Label Verifier - Simple Multi-GPU Script (No Copy Version)
# 원본 경로를 직접 사용 - 복사 없음
################################################################################

NUM_GPUS=${1:-$(nvidia-smi --list-gpus | wc -l)}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected!"
    exit 1
fi

echo "======================================================================"
echo "  YOLO Label Verifier - Multi-GPU Mode (No Copy)"
echo "  Using $NUM_GPUS GPUs"
echo "======================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python으로 이미지 리스트 가져오기
python3 << 'EOF' > image_list.txt
import yaml
from pathlib import Path

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

input_dir = Path(config['paths']['input_images'])

for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    for img_path in input_dir.glob(ext):
        print(img_path)
EOF

TOTAL_IMAGES=$(wc -l < image_list.txt)
echo "Total images: $TOTAL_IMAGES"

if [ "$TOTAL_IMAGES" -eq 0 ]; then
    echo "Error: No images found!"
    rm image_list.txt
    exit 1
fi

# 이미지 리스트를 GPU 개수만큼 분할
split -n l/$NUM_GPUS -d -a 2 image_list.txt split_

echo "Image list split complete"
echo ""

# 임시 Python 스크립트 생성 (각 GPU용) - 원본 경로 직접 사용
cat > run_gpu_worker_nocopy.py << 'EOFPYTHON'
import sys
import os
import yaml
from pathlib import Path

gpu_id = int(sys.argv[1])
image_list_file = sys.argv[2]

# Config 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 이미지 리스트 로드
with open(image_list_file, 'r', encoding='utf-8') as f:
    image_paths = [Path(line.strip()) for line in f if line.strip()]

print(f"[GPU {gpu_id}] Processing {len(image_paths)} images")

# Config 수정 (output만)
config['paths']['output_base'] = config['paths']['output_base'] + f'_gpu{gpu_id}'
config['processing']['show_progress'] = False

# 임시 config 저장
temp_dir = Path(f'temp_gpu{gpu_id}')
temp_dir.mkdir(exist_ok=True)
temp_config = temp_dir / 'config.yaml'

with open(temp_config, 'w', encoding='utf-8') as f:
    yaml.dump(config, f)

# verifier 실행
print(f"[GPU {gpu_id}] Starting verification...")
import verifier

import logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - [GPU {gpu_id}] %(levelname)s - %(message)s'
)

with open(temp_config, 'r', encoding='utf-8') as f:
    config_loaded = yaml.safe_load(f)

# ModelManager 실행
model_manager = verifier.ModelManager(config_loaded)

# LabelVerifier 실행
label_verifier = verifier.LabelVerifier(model_manager, config_loaded)

# ResultManager 실행
result_manager = verifier.ResultManager(config_loaded)

# 이미지 리스트에서 직접 처리
input_labels = Path(config_loaded['paths']['input_labels'])

results_log = []
processed_count = 0

for idx, image_path in enumerate(image_paths):
    label_path = input_labels / f"{image_path.stem}.txt"
    
    if not label_path.exists():
        logging.warning(f"Label not found: {label_path.name}")
        continue
    
    try:
        # 한글 경로 처리
        import numpy as np
        import cv2
        with open(image_path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            logging.error(f"Failed to load image: {image_path.name}")
            continue
        
        # Scene 분류
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        scene_info = verifier.SceneInfo("unknown", "unknown", 0.0)
        if config_loaded.get('scene_classification', {}).get('enabled', False):
            try:
                scene_info = label_verifier.classify_scene(pil_image)
            except Exception as e:
                logging.warning(f"Scene classification failed: {e}")
        
        # 검증
        category, box_results = label_verifier.verify_image(image_path, label_path)
        
        if box_results:
            boxes = verifier.YOLOParser.parse_label_file(label_path)
            result_manager.save_debug_visualization(
                image_path, img, boxes, box_results, 
                config_loaded['classes'], scene_info)
        
        # 파일 복사
        result_manager.copy_files(image_path, label_path, category, scene_info)
        
        # 수정된 라벨 저장
        if box_results:
            result_manager.save_corrected_label(
                label_path, boxes, box_results, category, scene_info)
        
        # 로그
        for result in box_results:
            results_log.append({
                'image': str(result.image_path),
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
        
        processed_count += 1
        
        if (idx + 1) % 10 == 0:
            msg = f"[GPU {gpu_id}] Progress: {idx + 1}/{len(image_paths)}"
            print(msg, flush=True)  # flush=True로 즉시 출력
            logging.info(msg)
            
            # 주기적으로 GPU 캐시 비우기
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        logging.error(f"Error processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()

# 리포트 저장
result_manager.save_report(results_log)
print(f"[GPU {gpu_id}] Complete! Processed {processed_count} images")

# 임시 파일 정리
import shutil
if temp_dir.exists():
    shutil.rmtree(temp_dir)
EOFPYTHON

# 각 GPU에서 백그라운드로 실행
echo "Starting verification on $NUM_GPUS GPUs..."
echo ""

for i in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_FILE="split_$(printf '%02d' $i)"
    
    if [ -f "$SPLIT_FILE" ]; then
        echo "[GPU $i] Starting..."
        CUDA_VISIBLE_DEVICES=$i python3 run_gpu_worker_nocopy.py $i $SPLIT_FILE > gpu${i}.log 2>&1 &
    fi
done

# 모든 GPU 작업 완료 대기
echo ""
echo "Waiting for all GPUs to complete..."
wait

echo ""
echo "======================================================================"
echo "  All GPUs completed!"
echo "======================================================================"
echo ""

# 결과 통합 (이전과 동일)
echo "Merging results..."

python3 << 'EOFMERGE'
import yaml
import json
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

output_base = Path(config['paths']['output_base'])
output_base.mkdir(parents=True, exist_ok=True)

import subprocess
result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
num_gpus = len([line for line in result.stdout.split('\n') if line.strip()])

categories = ['correct', 'mislabeled', 'uncertain']

print(f"Merging results from {num_gpus} GPUs...")

for gpu_id in range(num_gpus):
    gpu_output = Path(str(output_base) + f'_gpu{gpu_id}')
    
    if not gpu_output.exists():
        print(f"Warning: GPU {gpu_id} output not found")
        continue
    
    print(f"Merging GPU {gpu_id}...")
    
    # 각 category별로 복사
    for category in categories:
        src_folder = gpu_output / category
        dst_folder = output_base / category
        
        if not src_folder.exists():
            continue
        
        # JPEGImages, labels, labels_original 복사
        for subdir in ['JPEGImages', 'labels', 'labels_original']:
            src_sub = src_folder / subdir
            if src_sub.exists():
                dst_sub = dst_folder / subdir
                dst_sub.mkdir(parents=True, exist_ok=True)
                for f in src_sub.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst_sub / f.name)
    
    src_debug = gpu_output / 'debug_images'
    if src_debug.exists():
        dst_debug = output_base / 'debug_images'
        dst_debug.mkdir(exist_ok=True)
        for f in src_debug.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_debug / f.name)

combined_stats = {
    'total_images': 0,
    'total_boxes': 0,
    'correct': 0,
    'mislabeled': 0,
    'uncertain': 0,
    'failed': 0,
    'per_class': defaultdict(lambda: {'correct': 0, 'mislabeled': 0, 'uncertain': 0}),
    'per_scene': {
        'indoor': {'day': 0, 'night': 0, 'unknown': 0},
        'outdoor': {'day': 0, 'night': 0, 'unknown': 0},
        'unknown': {'day': 0, 'night': 0, 'unknown': 0}
    }
}

all_results = []

for gpu_id in range(num_gpus):
    report_file = Path(str(output_base) + f'_gpu{gpu_id}') / 'verification_report.json'
    
    if not report_file.exists():
        continue
    
    with open(report_file, 'r', encoding='utf-8') as f:
        gpu_report = json.load(f)
    
    stats = gpu_report['statistics']
    combined_stats['total_images'] += stats['total_images']
    combined_stats['total_boxes'] += stats['total_boxes']
    combined_stats['correct'] += stats['correct']
    combined_stats['mislabeled'] += stats['mislabeled']
    combined_stats['uncertain'] += stats['uncertain']
    combined_stats['failed'] += stats['failed']
    
    for class_name, class_stats in stats['per_class'].items():
        combined_stats['per_class'][class_name]['correct'] += class_stats['correct']
        combined_stats['per_class'][class_name]['mislabeled'] += class_stats['mislabeled']
        combined_stats['per_class'][class_name]['uncertain'] += class_stats['uncertain']
    
    for location, times_dict in stats['per_scene'].items():
        for time, count in times_dict.items():
            combined_stats['per_scene'][location][time] += count
    
    all_results.extend(gpu_report['detailed_results'])

combined_report = {
    'timestamp': datetime.now().isoformat(),
    'num_gpus': num_gpus,
    'statistics': {**combined_stats, 'per_class': dict(combined_stats['per_class'])},
    'detailed_results': all_results
}

with open(output_base / 'verification_report.json', 'w', encoding='utf-8') as f:
    json.dump(combined_report, f, indent=2, ensure_ascii=False)

print(f"\n✓ Results merged successfully!")
print(f"  Output: {output_base}")
print(f"  Total images: {combined_stats['total_images']}")
print(f"  Total boxes: {combined_stats['total_boxes']}")
print(f"  Correct: {combined_stats['correct']}")
print(f"  Mislabeled: {combined_stats['mislabeled']}")
print(f"  Uncertain: {combined_stats['uncertain']}")

print("\nCleaning up GPU outputs...")
for gpu_id in range(num_gpus):
    gpu_output = Path(str(output_base) + f'_gpu{gpu_id}')
    if gpu_output.exists():
        shutil.rmtree(gpu_output)
EOFMERGE

# 임시 파일 정리
rm -f image_list.txt
rm -f split_*
rm -f run_gpu_worker_nocopy.py

echo ""
echo "======================================================================"
echo "✓ Multi-GPU verification complete!"
echo "======================================================================"
