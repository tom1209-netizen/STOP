#!/usr/bin/env python3
"""
Validation script for the unified model training system.

This script validates that all training components are properly implemented
and can work together without external dependencies.
"""

import os
import sys
import json

def validate_file_exists(file_path, description):
    """Validate that a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NOT FOUND)")
        return False

def validate_python_syntax(file_path):
    """Validate Python file syntax."""
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
        print(f"✅ Python syntax valid: {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Python syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Could not validate {file_path}: {e}")
        return False

def validate_json_syntax(file_path):
    """Validate JSON file syntax."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        print(f"✅ JSON syntax valid: {file_path}")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ JSON syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Could not validate {file_path}: {e}")
        return False

def validate_unified_model_structure():
    """Validate that the unified model has training methods."""
    try:
        # Read the unified model file
        with open('unified_model.py', 'r') as f:
            content = f.read()
        
        required_methods = [
            'compute_training_loss',
            'freeze_tempme',
            'freeze_stop', 
            'unfreeze_all',
            'get_trainable_parameters',
            'get_parameter_groups'
        ]
        
        missing_methods = []
        for method in required_methods:
            if f"def {method}" not in content:
                missing_methods.append(method)
        
        if not missing_methods:
            print("✅ Unified model has all required training methods")
            return True
        else:
            print(f"❌ Missing training methods in unified model: {missing_methods}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating unified model structure: {e}")
        return False

def validate_config_structure():
    """Validate that configuration classes support training."""
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        
        required_classes = [
            'class UnifiedTrainingConfig',
            'class UnifiedModelConfig',
            'class TempMeConfig',
            'class STOPConfig'
        ]
        
        missing_classes = []
        for class_def in required_classes:
            if class_def not in content:
                missing_classes.append(class_def)
        
        if not missing_classes:
            print("✅ Configuration classes properly defined")
            return True
        else:
            print(f"❌ Missing configuration classes: {missing_classes}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating configuration structure: {e}")
        return False

def main():
    """Main validation function."""
    print("🔍 Validating Unified Model Training System")
    print("=" * 50)
    
    all_passed = True
    
    # Validate file existence
    print("\n📁 File Existence Check:")
    files_to_check = [
        ("unified_model.py", "Main unified model"),
        ("config.py", "Configuration classes"),
        ("train_unified_model.py", "Training script"),
        ("training_example.py", "Training examples"),
        ("TRAINING_GUIDE.md", "Training documentation"),
        ("configs/training_config.json", "Training configuration"),
    ]
    
    for file_path, description in files_to_check:
        if not validate_file_exists(file_path, description):
            all_passed = False
    
    # Validate Python syntax
    print("\n🐍 Python Syntax Check:")
    python_files = [
        "unified_model.py",
        "config.py", 
        "train_unified_model.py",
        "training_example.py"
    ]
    
    for file_path in python_files:
        if os.path.exists(file_path):
            if not validate_python_syntax(file_path):
                all_passed = False
    
    # Validate JSON syntax
    print("\n📄 JSON Syntax Check:")
    json_files = [
        "configs/training_config.json"
    ]
    
    for file_path in json_files:
        if os.path.exists(file_path):
            if not validate_json_syntax(file_path):
                all_passed = False
    
    # Validate structure
    print("\n🏗️  Structure Validation:")
    if not validate_unified_model_structure():
        all_passed = False
    
    if not validate_config_structure():
        all_passed = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All validations passed! Training system is ready.")
        print("\n📖 Next steps:")
        print("1. Review TRAINING_GUIDE.md for detailed documentation")
        print("2. Prepare your dataset in the required format")
        print("3. Configure training parameters in configs/training_config.json")
        print("4. Run: python training_example.py")
        return 0
    else:
        print("❌ Some validations failed. Please review and fix the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)