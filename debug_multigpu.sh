#!/bin/bash

################################################################################
# Multi-GPU Debugging Script
# 멀티 GPU 실행 실패 원인 진단
################################################################################

echo "======================================================================"
echo "  Multi-GPU Debugging Tool"
echo "======================================================================"
echo ""

# 색상
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SUCCESS=0
FAILED=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}✗${NC} $1"
        FAILED=$((FAILED + 1))
    fi
}

# 1. Log files
echo "[1] Checking log files..."
if ls gpu*.log 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Log files exist:"
    ls -lh gpu*.log
    echo ""
    echo "Last 30 lines of GPU 0 log:"
    echo "---"
    tail -30 gpu0.log
    echo "---"
    echo ""
    
    # 에러 검색
    echo "Errors in logs:"
    grep -i "error\|exception\|failed\|traceback" gpu*.log | head -20
else
    echo -e "${RED}✗${NC} No log files found"
    echo "  This means GPUs didn't start or crashed immediately"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "[2] Checking split files..."
if ls split_* 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Split files exist:"
    wc -l split_*
    echo ""
    echo "Sample from split_00:"
    head -3 split_00
else
    echo -e "${RED}✗${NC} No split files found"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "[3] Checking temp directories..."
if ls -d temp_gpu* 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Temp directories exist:"
    for dir in temp_gpu*/; do
        if [ -d "$dir" ]; then
            echo "  $dir"
            if [ -d "${dir}JPEGImages" ]; then
                img_count=$(ls -1 ${dir}JPEGImages/ 2>/dev/null | wc -l)
                echo "    Images: $img_count"
                
                # Symlink 체크
                if [ $img_count -gt 0 ]; then
                    first_file="${dir}JPEGImages/$(ls -1 ${dir}JPEGImages/ | head -1)"
                    if [ -L "$first_file" ]; then
                        echo -e "    Type: ${YELLOW}Symlink${NC}"
                        if [ -e "$first_file" ]; then
                            echo -e "    Status: ${GREEN}Valid${NC}"
                        else
                            echo -e "    Status: ${RED}Broken!${NC}"
                        fi
                    else
                        echo -e "    Type: ${GREEN}Real file${NC}"
                    fi
                fi
            fi
        fi
    done
else
    echo -e "${YELLOW}○${NC} No temp directories (might have been cleaned up)"
fi

echo ""
echo "[4] Checking Python and dependencies..."
python3 << 'EOFPY'
import sys
print(f"Python version: {sys.version.split()[0]}")

# Check imports
modules = ['yaml', 'cv2', 'torch', 'PIL', 'numpy', 'transformers']
for mod in modules:
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except ImportError:
        print(f"  ✗ {mod} - NOT INSTALLED")

# Check verifier
try:
    import verifier
    print(f"  ✓ verifier.py")
except Exception as e:
    print(f"  ✗ verifier.py - {e}")
EOFPY

echo ""
echo "[5] Checking config.yaml..."
python3 << 'EOFPY'
import yaml
from pathlib import Path

try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("✓ config.yaml is valid")
    
    # Check paths
    input_images = Path(config['paths']['input_images'])
    input_labels = Path(config['paths']['input_labels'])
    output_base = Path(config['paths']['output_base'])
    
    print(f"\nPaths:")
    print(f"  Input images: {input_images}")
    if input_images.exists():
        img_count = len(list(input_images.glob('*.jpg')) + 
                        list(input_images.glob('*.jpeg')) + 
                        list(input_images.glob('*.png')))
        print(f"    ✓ Exists ({img_count} images)")
    else:
        print(f"    ✗ NOT FOUND")
    
    print(f"  Input labels: {input_labels}")
    if input_labels.exists():
        label_count = len(list(input_labels.glob('*.txt')))
        print(f"    ✓ Exists ({label_count} labels)")
    else:
        print(f"    ✗ NOT FOUND")
    
    print(f"  Output base: {output_base}")
    if output_base.exists():
        print(f"    ✓ Exists")
    else:
        print(f"    ○ Will be created")
    
    # Check classes
    print(f"\nClasses: {len(config.get('classes', {}))} defined")
    
except Exception as e:
    print(f"✗ config.yaml error: {e}")
EOFPY

echo ""
echo "[6] Checking model..."
if [ -d "./models" ]; then
    echo -e "${GREEN}✓${NC} Models directory exists"
    model_count=$(find ./models -name "*.safetensors" -o -name "*.bin" | wc -l)
    if [ $model_count -gt 0 ]; then
        echo "  Model files: $model_count"
    else
        echo -e "  ${YELLOW}○${NC} No model files (will download on first run)"
    fi
else
    echo -e "${YELLOW}○${NC} Models directory not found (will be created)"
fi

echo ""
echo "[7] Checking GPU outputs..."
python3 << 'EOFPY'
import yaml
from pathlib import Path
import subprocess

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

output_base = Path(config['paths']['output_base'])

# Detect GPU count
try:
    result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                          capture_output=True, text=True)
    num_gpus = len([l for l in result.stdout.split('\n') if l.strip()])
except:
    num_gpus = 8

print(f"Expected GPU outputs: {num_gpus}")

found = 0
for gpu_id in range(num_gpus):
    gpu_output = Path(str(output_base) + f'_gpu{gpu_id}')
    if gpu_output.exists():
        print(f"  ✓ GPU {gpu_id} output exists: {gpu_output}")
        
        # Check report
        report = gpu_output / 'verification_report.json'
        if report.exists():
            import json
            with open(report) as f:
                data = json.load(f)
                stats = data.get('statistics', {})
                print(f"      Images: {stats.get('total_images', 0)}, "
                      f"Boxes: {stats.get('total_boxes', 0)}")
        found += 1
    else:
        print(f"  ✗ GPU {gpu_id} output NOT found")

if found == 0:
    print("\n  ⚠ No GPU outputs found - this is the main problem!")
EOFPY

echo ""
echo "======================================================================"
echo "  Diagnosis Summary"
echo "======================================================================"
echo -e "Checks passed: ${GREEN}$SUCCESS${NC}"
echo -e "Checks failed: ${RED}$FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Common fixes:"
    echo "  1. Check GPU logs: cat gpu0.log"
    echo "  2. Try without symlinks: bash run_multigpu_nocopy.sh"
    echo "  3. Test single GPU first: python verifier.py"
    echo "  4. Check file permissions: chmod -R 755 ."
fi

echo "======================================================================"
