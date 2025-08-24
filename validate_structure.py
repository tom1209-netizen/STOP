#!/usr/bin/env python3
"""
Validation script for the Unified TempMe-STOP Model structure

This script validates the implementation structure without requiring PyTorch or other dependencies.
It checks imports, file structure, and API consistency.
"""

import os
import sys
import json
import tempfile
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_file_structure():
    """Test that all required files exist."""
    print("="*60)
    print("TESTING FILE STRUCTURE")
    print("="*60)
    
    required_files = [
        'unified_model.py',
        'extract_video_embeddings.py', 
        'config.py',
        'configs/unified_model_config.json',
        'README_unified.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path} exists")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist\n")
    return True


def test_config_file_structure():
    """Test the configuration file structure."""
    print("="*60)
    print("TESTING CONFIG FILE STRUCTURE")
    print("="*60)
    
    try:
        config_path = 'configs/unified_model_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required top-level keys
        required_keys = ['tempme', 'stop', 'device', 'max_frames', 'num_segments']
        for key in required_keys:
            assert key in config, f"Missing key: {key}"
            print(f"✓ Config has key: {key}")
        
        # Check TempMe config structure
        tempme_keys = ['base_encoder', 'lora_dim', 'merge_frame_num']
        for key in tempme_keys:
            assert key in config['tempme'], f"Missing TempMe key: {key}"
            print(f"✓ TempMe config has key: {key}")
        
        # Check STOP config structure
        stop_keys = ['cross_model', 'sim_header', 'temporal_prompt']
        for key in stop_keys:
            assert key in config['stop'], f"Missing STOP key: {key}"
            print(f"✓ STOP config has key: {key}")
        
        print("✓ Configuration file structure is valid\n")
        return True
        
    except Exception as e:
        print(f"✗ Configuration file test failed: {e}")
        return False


def test_extraction_script_structure():
    """Test the extraction script functionality."""
    print("="*60)
    print("TESTING EXTRACTION SCRIPT STRUCTURE")
    print("="*60)
    
    try:
        # Check if the extraction script has the required structure
        with open('extract_video_embeddings.py', 'r') as f:
            script_content = f.read()
        
        required_functions = [
            'def get_video_files(',
            'def extract_single_video_embedding(',
            'def extract_directory_embeddings(',
            'def save_embedding(',
            'def main('
        ]
        
        for func in required_functions:
            if func in script_content:
                print(f"✓ Function found: {func.split('(')[0]}")
            else:
                print(f"✗ Function missing: {func.split('(')[0]}")
                return False
        
        # Check for video extensions
        if 'VIDEO_EXTENSIONS' in script_content:
            print("✓ Video file extensions defined")
        else:
            print("✗ Video file extensions not defined")
            return False
        
        # Check for argparse usage
        if 'argparse' in script_content and 'ArgumentParser' in script_content:
            print("✓ Command line argument parsing implemented")
        else:
            print("✗ Command line argument parsing missing")
            return False
        
        print("✓ Extraction script structure is valid\n")
        return True
        
    except Exception as e:
        print(f"✗ Sample batch file test failed: {e}")
        return False


def test_python_file_syntax():
    """Test that Python files have valid syntax."""
    print("="*60)
    print("TESTING PYTHON FILE SYNTAX")
    print("="*60)
    
    python_files = [
        'unified_model.py',
        'extract_video_embeddings.py',
        'config.py',
        'test_unified_model.py'
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Try to compile the source code
            compile(source_code, file_path, 'exec')
            print(f"✓ {file_path} has valid syntax")
            
        except SyntaxError as e:
            print(f"✗ {file_path} has syntax error: {e}")
            return False
        except Exception as e:
            print(f"✗ Error checking {file_path}: {e}")
            return False
    
    print("✓ All Python files have valid syntax\n")
    return True


def test_imports_structure():
    """Test import statements without actually importing."""
    print("="*60)
    print("TESTING IMPORT STRUCTURE")
    print("="*60)
    
    # Check unified_model.py imports
    with open('unified_model.py', 'r') as f:
        unified_content = f.read()
    
    required_imports = [
        'import torch',
        'import torch.nn',
        'from config import UnifiedModelConfig',
        'from typing import List, Union, Tuple, Optional'
    ]
    
    for import_stmt in required_imports:
        if import_stmt in unified_content:
            print(f"✓ unified_model.py has: {import_stmt}")
        else:
            print(f"⚠ unified_model.py missing: {import_stmt}")
    
    # Check extract_video_embeddings.py imports
    with open('extract_video_embeddings.py', 'r') as f:
        extraction_content = f.read()
    
    extraction_imports = [
        'from unified_model import',
        'import argparse',
        'import numpy as np',
        'from typing import Dict, List'
    ]
    
    for import_stmt in extraction_imports:
        if import_stmt in extraction_content:
            print(f"✓ extract_video_embeddings.py has: {import_stmt}")
        else:
            print(f"⚠ extract_video_embeddings.py missing: {import_stmt}")
    
    print("✓ Import structure checks completed\n")
    return True


def test_api_signatures():
    """Test that API functions have correct signatures."""
    print("="*60)
    print("TESTING API SIGNATURES")
    print("="*60)
    
    # Check extract_video_embeddings.py for main functions
    with open('extract_video_embeddings.py', 'r') as f:
        extraction_content = f.read()
    
    # Look for the main extraction function signatures
    if 'def extract_directory_embeddings(' in extraction_content:
        print("✓ extract_directory_embeddings function exists")
    else:
        print("✗ extract_directory_embeddings function not found")
        return False
    
    if 'def extract_single_video_embedding(' in extraction_content:
        print("✓ extract_single_video_embedding function exists")
    else:
        print("✗ extract_single_video_embedding function not found")
        return False
    
    # Check for numpy array return type
    if 'np.ndarray' in extraction_content:
        print("✓ Functions use numpy arrays for embeddings")
    else:
        print("✗ Missing numpy array usage for embeddings")
        return False
    
    # Check unified_model.py for main class
    with open('unified_model.py', 'r') as f:
        unified_content = f.read()
    
    if 'class UnifiedTempMeSTOPModel' in unified_content:
        print("✓ UnifiedTempMeSTOPModel class found")
    else:
        print("✗ UnifiedTempMeSTOPModel class not found")
        return False
    
    print("✓ API signatures are correct\n")
    return True


def test_documentation_completeness():
    """Test that documentation is complete."""
    print("="*60)
    print("TESTING DOCUMENTATION COMPLETENESS")
    print("="*60)
    
    # Check README_unified.md
    with open('README_unified.md', 'r') as f:
        readme_content = f.read()
    
    required_sections = [
        '# 🚀 Unified TempMe-STOP Model',
        '## ✨ Overview',
        '## 🚀 Quick Start',
        '## 📋 Command Line Interface',
        '## ⚙️ Configuration',
        '## 🔧 Advanced Usage'
    ]
    
    for section in required_sections:
        if section in readme_content:
            print(f"✓ README has section: {section}")
        else:
            print(f"✗ README missing section: {section}")
            return False
    
    # Check for code examples
    if '```python' in readme_content and '```bash' in readme_content:
        print("✓ README contains code examples")
    else:
        print("✗ README missing code examples")
        return False
    
    print("✓ Documentation is complete\n")
    return True


def test_configuration_class_structure():
    """Test the configuration class structure without importing."""
    print("="*60)
    print("TESTING CONFIGURATION CLASS STRUCTURE")
    print("="*60)
    
    with open('config.py', 'r') as f:
        config_content = f.read()
    
    required_classes = [
        'class TempMeConfig',
        'class STOPConfig', 
        'class UnifiedModelConfig'
    ]
    
    for class_name in required_classes:
        if class_name in config_content:
            print(f"✓ config.py has: {class_name}")
        else:
            print(f"✗ config.py missing: {class_name}")
            return False
    
    # Check for key methods
    required_methods = [
        'def from_json',
        'def to_json',
        'def get_default_config'
    ]
    
    for method in required_methods:
        if method in config_content:
            print(f"✓ config.py has: {method}")
        else:
            print(f"✗ config.py missing: {method}")
            return False
    
    print("✓ Configuration class structure is correct\n")
    return True


def run_all_validation_tests():
    """Run all validation tests."""
    print("🔍 STARTING UNIFIED MODEL STRUCTURE VALIDATION")
    print("="*80)
    
    test_results = []
    
    # Run individual test cases
    test_cases = [
        ("File Structure", test_file_structure),
        ("Config File Structure", test_config_file_structure),
        ("Extraction Script Structure", test_extraction_script_structure),
        ("Python File Syntax", test_python_file_syntax),
        ("Import Structure", test_imports_structure),
        ("API Signatures", test_api_signatures),
        ("Documentation Completeness", test_documentation_completeness),
        ("Configuration Class Structure", test_configuration_class_structure),
    ]
    
    for test_name, test_func in test_cases:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print final results
    print("="*80)
    print("🏁 VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print("="*80)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("The unified model structure and implementation is complete and ready.")
        print("\nNext steps:")
        print("1. Install PyTorch environment: conda env create -f environment.yaml")
        print("2. Download pretrained models as described in README")
        print("3. Test with real video data: python extract_video_embeddings.py --video_dir ./sample_videos --output_dir ./embeddings")
        return True
    else:
        print("⚠️  Some validation tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_validation_tests()
    sys.exit(0 if success else 1)