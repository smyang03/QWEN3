#!/bin/bash

################################################################################
# YOLO Label Verifier - Multi-GPU (Progress Visible)
# 진행 상황 화면 표시 + 단순 폴더 구조
################################################################################

NUM_GPUS=${1:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected!"
    exit 1
fi

echo "======================================================================"
echo "  YOLO Label Verifier - Multi-GPU Mode"
echo "  Using $NUM_GPUS GPUs"
echo "======================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 이미지 리스트 생성
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

# 이미지 리스트 분할
split -n l/$NUM_GPUS -d -a 2 image_list.txt split_

echo "Image list split complete"
echo ""

# GPU Worker Python 스크립트 생성
cat > run_gpu_worker_v2.py << 'EOFPYTHON'
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime

gpu_id = int(sys.argv[1])
image_list_file = sys.argv[2]

# 오프라인 모드 설정 (Hugging Face)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# Config 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 이미지 리스트 로드
with open(image_list_file, 'r', encoding='utf-8') as f:
    image_paths = [Path(line.strip()) for line in f if line.strip()]

print(f"[GPU {gpu_id}] Processing {len(image_paths)} images", flush=True)

# Progress 출력 주기 설정
progress_interval = max(1, len(image_paths) // 100)  # 최소 1, 최대 1%마다

# verifier 모듈 임포트
import verifier
import logging

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - [GPU {gpu_id}] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gpu{gpu_id}.log'),
        logging.StreamHandler()  # 화면에도 출력
    ]
)

# ModelManager 실행
model_manager = verifier.ModelManager(config)

# LabelVerifier 실행
label_verifier = verifier.LabelVerifier(model_manager, config)

# ResultManager 실행
result_manager = verifier.ResultManager(config)

# 검증 실행
input_labels = Path(config['paths']['input_labels'])

results_log = []
processed_count = 0
start_time = datetime.now()

for idx, image_path in enumerate(image_paths):
    label_path = input_labels / f"{image_path.stem}.txt"
    
    if not label_path.exists():
        logging.warning(f"Label not found: {label_path.name}")
        continue
    
    try:
        # 이미지 로드
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
        if config.get('scene_classification', {}).get('enabled', False):
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
                config['classes'], scene_info)
        
        # 파일 복사 (scene_info는 무시됨 - 단순 구조)
        result_manager.copy_files(image_path, label_path, category, scene_info)
        
        # 수정된 라벨 저장
        if box_results:
            result_manager.save_corrected_label(
                label_path, boxes, box_results, category, scene_info)
        
        # 로그 수집
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
        
        # Progress 출력 (화면에도 표시)
        if (idx + 1) % progress_interval == 0 or (idx + 1) == len(image_paths):
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = processed_count / elapsed if elapsed > 0 else 0
            remaining = (len(image_paths) - processed_count) / speed if speed > 0 else 0
            
            progress_msg = (
                f"[GPU {gpu_id}] Progress: {idx + 1}/{len(image_paths)} "
                f"({100*(idx+1)/len(image_paths):.1f}%) | "
                f"Speed: {speed:.1f} img/s | "
                f"ETA: {remaining/60:.1f} min"
            )
            print(progress_msg, flush=True)
            
            # GPU 메모리 정리
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        logging.error(f"Error processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()

# 리포트 저장 (GPU별 임시 파일)
temp_report = Path(f'temp_report_gpu{gpu_id}.json')
import json
with open(temp_report, 'w', encoding='utf-8') as f:
    json.dump({
        'gpu_id': gpu_id,
        'processed': processed_count,
        'results': results_log
    }, f, indent=2, ensure_ascii=False)

print(f"[GPU {gpu_id}] ✓ Complete! Processed {processed_count} images", flush=True)
EOFPYTHON

# GPU 작업 시작
echo "Starting verification on $NUM_GPUS GPUs..."
echo ""
echo "Progress will be displayed below:"
echo "======================================================================"
echo ""

pids=()

for i in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_FILE="split_$(printf '%02d' $i)"
    
    if [ -f "$SPLIT_FILE" ]; then
        # tee로 화면과 파일 동시 출력
        python3 run_gpu_worker_v2.py $i $SPLIT_FILE 2>&1 | tee gpu${i}.log &
        pids+=($!)
    fi
done

# 모든 GPU 완료 대기
wait "${pids[@]}"

echo ""
echo "======================================================================"
echo "  All GPUs completed!"
echo "======================================================================"
echo ""

# 결과 통합
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

# GPU별 임시 리포트 로드
all_results = []
total_processed = 0

import subprocess
result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
num_gpus = len([line for line in result.stdout.split('\n') if line.strip()])

for gpu_id in range(num_gpus):
    temp_report = Path(f'temp_report_gpu{gpu_id}.json')
    if temp_report.exists():
        with open(temp_report, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_results.extend(data['results'])
            total_processed += data['processed']

print(f"Merged {total_processed} images from {num_gpus} GPUs")

# 통계 계산
stats = {
    'total_images': total_processed,
    'total_boxes': len(all_results),
    'correct': sum(1 for r in all_results if r['is_correct']),
    'mislabeled': sum(1 for r in all_results if not r['is_correct'] and r['category'] == 'mislabeled'),
    'uncertain': sum(1 for r in all_results if r['category'] == 'uncertain'),
    'per_class': defaultdict(lambda: {'correct': 0, 'mislabeled': 0, 'uncertain': 0}),
    'per_scene': {
        'indoor': {'day': 0, 'night': 0, 'unknown': 0},
        'outdoor': {'day': 0, 'night': 0, 'unknown': 0},
        'unknown': {'day': 0, 'night': 0, 'unknown': 0}
    }
}

for r in all_results:
    class_name = r['class_name']
    if r['is_correct']:
        stats['per_class'][class_name]['correct'] += 1
    elif r['category'] == 'mislabeled':
        stats['per_class'][class_name]['mislabeled'] += 1
    else:
        stats['per_class'][class_name]['uncertain'] += 1
    
    stats['per_scene'][r['scene_location']][r['scene_time']] += 1

# 최종 리포트 저장
final_report = {
    'timestamp': datetime.now().isoformat(),
    'num_gpus': num_gpus,
    'statistics': {**stats, 'per_class': dict(stats['per_class'])},
    'detailed_results': all_results
}

report_file = output_base / 'verification_report.json'
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

print(f"\n✓ Results merged successfully!")
print(f"  Output: {output_base}")
print(f"  Total images: {stats['total_images']}")
print(f"  Total boxes: {stats['total_boxes']}")
print(f"  Correct: {stats['correct']}")
print(f"  Mislabeled: {stats['mislabeled']}")
print(f"  Uncertain: {stats['uncertain']}")

# 임시 파일 정리
print("\nCleaning up temporary files...")
for gpu_id in range(num_gpus):
    temp_report = Path(f'temp_report_gpu{gpu_id}.json')
    if temp_report.exists():
        temp_report.unlink()
EOFMERGE

# 임시 파일 정리
rm -f image_list.txt
rm -f split_*
rm -f run_gpu_worker_v2.py

echo ""
echo "======================================================================"
echo "✓ Multi-GPU verification complete!"
echo "======================================================================"
