#!/bin/bash

################################################################################
# Offline Mode Test Script
# 오프라인 환경 설정 확인
################################################################################

echo "======================================================================"
echo "  Offline Mode Check"
echo "======================================================================"
echo ""

# 1. verifier.py 확인
echo "[1] Checking verifier.py for local_files_only..."
if grep -q "local_files_only=True" verifier.py; then
    count=$(grep -c "local_files_only=True" verifier.py)
    echo "  ✓ Found $count occurrences of local_files_only=True"
else
    echo "  ✗ local_files_only=True NOT FOUND"
    echo "  Fix: Add 'local_files_only=True' to from_pretrained() calls"
fi

echo ""

# 2. 환경 변수 확인
echo "[2] Checking environment variables..."
if [ -n "$HF_HUB_OFFLINE" ]; then
    echo "  ✓ HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
else
    echo "  ○ HF_HUB_OFFLINE not set (will be set by script)"
fi

if [ -n "$TRANSFORMERS_OFFLINE" ]; then
    echo "  ✓ TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
else
    echo "  ○ TRANSFORMERS_OFFLINE not set (will be set by script)"
fi

echo ""

# 3. 모델 파일 확인
echo "[3] Checking model files..."

MODEL_DIR="models/models--Qwen--Qwen3-VL-8B-Instruct"

if [ -d "$MODEL_DIR" ]; then
    echo "  ✓ Model directory exists: $MODEL_DIR"
    
    # config.json 확인
    if find "$MODEL_DIR" -name "config.json" | grep -q .; then
        config_path=$(find "$MODEL_DIR" -name "config.json" | head -1)
        echo "  ✓ config.json found: $config_path"
    else
        echo "  ✗ config.json NOT FOUND"
        echo "    Model may not be fully downloaded"
    fi
    
    # model 파일 확인
    if find "$MODEL_DIR" -name "*.safetensors" | grep -q .; then
        model_count=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)
        echo "  ✓ Model files found: $model_count .safetensors files"
    else
        echo "  ✗ Model files (.safetensors) NOT FOUND"
        echo "    Model may not be fully downloaded"
    fi
    
else
    echo "  ✗ Model directory NOT FOUND: $MODEL_DIR"
    echo "    Please download the model first"
fi

echo ""

# 4. Python 패키지 확인
echo "[4] Checking Python packages..."
python3 << 'EOFPY'
try:
    import transformers
    print(f"  ✓ transformers version: {transformers.__version__}")
except ImportError:
    print("  ✗ transformers not installed")

try:
    import torch
    print(f"  ✓ torch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA devices: {torch.cuda.device_count()}")
except ImportError:
    print("  ✗ torch not installed")
EOFPY

echo ""

# 5. 네트워크 상태 확인
echo "[5] Checking network connectivity..."
if ping -c 1 -W 1 huggingface.co &> /dev/null; then
    echo "  ✓ Can reach huggingface.co (online mode possible)"
    echo "  Note: Offline mode will still be used if configured"
else
    echo "  ✗ Cannot reach huggingface.co (offline mode required)"
    echo "  This is expected in offline environments"
fi

echo ""
echo "======================================================================"
echo "  Summary"
echo "======================================================================"
echo ""

# 전체 평가
if grep -q "local_files_only=True" verifier.py && \
   [ -d "$MODEL_DIR" ] && \
   find "$MODEL_DIR" -name "config.json" | grep -q . && \
   find "$MODEL_DIR" -name "*.safetensors" | grep -q .; then
    echo "✓ Ready for offline mode!"
    echo ""
    echo "Run with:"
    echo "  export HF_HUB_OFFLINE=1"
    echo "  export TRANSFORMERS_OFFLINE=1"
    echo "  bash run_multigpu_v2.sh"
else
    echo "✗ Not ready for offline mode"
    echo ""
    echo "Issues to fix:"
    
    if ! grep -q "local_files_only=True" verifier.py; then
        echo "  1. Add local_files_only=True to verifier.py"
    fi
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "  2. Download model to $MODEL_DIR"
    fi
    
    if ! find "$MODEL_DIR" -name "config.json" | grep -q .; then
        echo "  3. Ensure config.json exists in model directory"
    fi
    
    if ! find "$MODEL_DIR" -name "*.safetensors" | grep -q .; then
        echo "  4. Ensure model files (.safetensors) exist"
    fi
fi

echo ""
echo "======================================================================"
