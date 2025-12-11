#!/usr/bin/env python3
"""
멀티 GPU 환경에서 동의어 매칭 테스트
각 GPU에서 독립적으로 config를 로드하고 동의어가 올바르게 작동하는지 확인
"""

import os
import yaml
import multiprocessing
from pathlib import Path
from typing import Tuple, List


def test_alias_on_gpu(gpu_id: int, config_path: Path) -> Tuple[int, bool, str]:
    """단일 GPU에서 동의어 테스트"""
    
    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # Config 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Classes 파싱 (verifier.py의 LabelVerifier.__init__과 동일)
        raw_classes = config['classes']
        class_map = {}
        class_aliases = {}
        
        for class_id, names in raw_classes.items():
            class_id = int(class_id)
            
            # 리스트 또는 단일 문자열 지원
            if isinstance(names, list):
                name_list = names
            else:
                name_list = [names]
            
            # 첫 번째 이름이 대표 이름
            primary_name = name_list[0]
            class_map[class_id] = primary_name
            
            # 모든 동의어 등록
            for name in name_list:
                class_aliases[name.lower()] = (class_id, primary_name)
        
        # 테스트 케이스
        test_cases = [
            ("hard hat", 3, "safety helmet"),
            ("automobile", 6, "car"),
            ("people", 0, "person"),
            ("pigeon", 13, "bird"),
        ]
        
        failed = []
        for vlm_response, expected_id, expected_name in test_cases:
            detected_lower = vlm_response.lower().strip()
            
            if detected_lower in class_aliases:
                matched_id, matched_name = class_aliases[detected_lower]
                if matched_id != expected_id or matched_name != expected_name:
                    failed.append(f"{vlm_response}: expected {expected_name}, got {matched_name}")
            else:
                failed.append(f"{vlm_response}: not found in aliases")
        
        if failed:
            return gpu_id, False, "; ".join(failed)
        else:
            return gpu_id, True, f"All {len(test_cases)} tests passed"
    
    except Exception as e:
        return gpu_id, False, f"Exception: {str(e)}"


def main():
    """멀티 GPU 동의어 테스트 메인"""
    
    print("=" * 70)
    print("멀티 GPU 동의어 매칭 테스트")
    print("=" * 70)
    print()
    
    # GPU 개수 확인
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True,
            text=True
        )
        num_gpus = len([line for line in result.stdout.split('\n') if line.strip()])
        
        if num_gpus == 0:
            print("Warning: No CUDA GPUs detected, testing with CPU")
            num_gpus = 1
    except Exception as e:
        print(f"Warning: Could not detect GPUs ({e}), testing with CPU")
        num_gpus = 1
    
    print(f"Testing on {num_gpus} GPU(s)")
    print()
    
    # Config 경로
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # 각 GPU에서 병렬 테스트
    if num_gpus > 1:
        with multiprocessing.Pool(processes=num_gpus) as pool:
            tasks = [(gpu_id, config_path) for gpu_id in range(num_gpus)]
            results = pool.starmap(test_alias_on_gpu, tasks)
    else:
        # Single GPU/CPU
        results = [test_alias_on_gpu(0, config_path)]
    
    # 결과 출력
    print("[Test Results]")
    print()
    
    all_passed = True
    for gpu_id, success, message in results:
        status = "✓" if success else "✗"
        print(f"  GPU {gpu_id}: {status} {message}")
        if not success:
            all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✓ All GPUs passed alias matching tests!")
        print("=" * 70)
        return 0
    else:
        print("✗ Some GPUs failed alias matching tests")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
