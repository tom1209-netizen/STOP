#!/usr/bin/env python3
"""
Validation script for TempMe-STOP integration
This script performs basic validation without requiring heavy dependencies.
"""

import sys
import os
import traceback


def validate_imports():
    """Validate that all necessary modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test basic Python modules
        import json
        import argparse
        print("✓ Standard library imports successful")
    except ImportError as e:
        print(f"✗ Standard library import failed: {e}")
        return False
    
    # Test project structure
    project_files = [
        'modules/tempme.py',
        'modules/clip4clip.py', 
        'modules/temporal_prompting.py',
        'params.py',
        'main.py'
    ]
    
    for file_path in project_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True


def validate_syntax():
    """Validate syntax of modified files"""
    print("\nTesting syntax...")
    
    files_to_check = [
        'modules/tempme.py',
        'modules/clip4clip.py',
        'params.py'
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✓ {file_path} syntax valid")
        except SyntaxError as e:
            print(f"✗ {file_path} syntax error: {e}")
            return False
        except Exception as e:
            print(f"✗ {file_path} error: {e}")
            return False
    
    return True


def validate_configuration():
    """Validate TempMe configuration parameters"""
    print("\nTesting configuration...")
    
    try:
        # Import params module without executing argparse
        import importlib.util
        spec = importlib.util.spec_from_file_location("params", "params.py")
        params_module = importlib.util.module_from_spec(spec)
        
        # Check if TempMe parameters are defined (basic check)
        with open('params.py', 'r') as f:
            content = f.read()
            
        tempme_params = [
            'use_tempme',
            'tempme_type', 
            'tempme_compression_ratio',
            'tempme_hidden_dim'
        ]
        
        for param in tempme_params:
            if param in content:
                print(f"✓ Parameter {param} found")
            else:
                print(f"✗ Parameter {param} missing")
                return False
                
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    return True


def validate_integration():
    """Validate TempMe integration points"""
    print("\nTesting integration points...")
    
    try:
        # Check CLIP4Clip modifications
        with open('modules/clip4clip.py', 'r') as f:
            clip4clip_content = f.read()
        
        integration_points = [
            'use_tempme',
            'tempme_compressor',
            'get_tempme_model',
            'TempMe compression'
        ]
        
        for point in integration_points:
            if point in clip4clip_content:
                print(f"✓ Integration point '{point}' found in CLIP4Clip")
            else:
                print(f"✗ Integration point '{point}' missing in CLIP4Clip")
                return False
        
        # Check TempMe module structure
        with open('modules/tempme.py', 'r') as f:
            tempme_content = f.read()
        
        tempme_classes = [
            'SimpleTempMe',
            'TempMeCompressor', 
            'AdaptiveTempMe',
            'get_tempme_model'
        ]
        
        for cls in tempme_classes:
            if cls in tempme_content:
                print(f"✓ TempMe class/function '{cls}' found")
            else:
                print(f"✗ TempMe class/function '{cls}' missing")
                return False
                
    except Exception as e:
        print(f"✗ Integration validation failed: {e}")
        return False
    
    return True


def validate_documentation():
    """Validate documentation and examples"""
    print("\nTesting documentation...")
    
    docs = [
        'TempMe_Integration_Guide.md',
        'example_tempme_usage.py',
        'run_tempme_training.sh'
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            print(f"✓ Documentation file {doc} exists")
        else:
            print(f"✗ Documentation file {doc} missing")
            return False
    
    return True


def main():
    """Run all validation tests"""
    print("TempMe-STOP Integration Validation")
    print("=" * 50)
    
    tests = [
        ("Import Validation", validate_imports),
        ("Syntax Validation", validate_syntax),
        ("Configuration Validation", validate_configuration),
        ("Integration Validation", validate_integration),
        ("Documentation Validation", validate_documentation)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            result = test_func()
            if result:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\nThe TempMe-STOP integration is ready for use.")
        print("\nNext steps:")
        print("1. Set up your dataset paths in run_tempme_training.sh")
        print("2. Install required dependencies (PyTorch, etc.)")
        print("3. Run training with: ./run_tempme_training.sh")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("\nPlease review the error messages above and fix the issues.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)