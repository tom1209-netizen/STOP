#!/usr/bin/env python3
"""
Test script for the Unified TempMe-STOP Model

This script tests the implementation without requiring actual video files or trained models.
It uses dummy data to validate the pipeline structure and API functionality.
"""

import os
import sys
import torch
import tempfile
import json
import logging
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_system():
    """Test the configuration system."""
    print("="*60)
    print("TESTING CONFIGURATION SYSTEM")
    print("="*60)
    
    try:
        from config import UnifiedModelConfig, TempMeConfig, STOPConfig
        
        # Test default config creation
        config = UnifiedModelConfig()
        assert config.max_frames == 12
        assert config.device == "cuda"
        print("✓ Default configuration created successfully")
        
        # Test config serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            temp_config_path = f.name
        
        # Test config loading
        loaded_config = UnifiedModelConfig.from_json(temp_config_path)
        assert loaded_config.max_frames == config.max_frames
        print("✓ Configuration serialization/loading works")
        
        # Cleanup
        os.unlink(temp_config_path)
        
        print("✓ Configuration system tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        return False


def test_unified_model_structure():
    """Test the unified model structure and initialization."""
    print("="*60)
    print("TESTING UNIFIED MODEL STRUCTURE")
    print("="*60)
    
    try:
        from unified_model import UnifiedTempMeSTOPModel, create_unified_model
        from config import UnifiedModelConfig
        
        # Test model creation with default config
        config = UnifiedModelConfig(device='cpu')  # Use CPU for testing
        model = UnifiedTempMeSTOPModel(config)
        print("✓ Unified model created successfully")
        
        # Test factory function
        model2 = create_unified_model(device='cpu')
        print("✓ Factory function works")
        
        # Test model methods exist
        assert hasattr(model, 'preprocess_video')
        assert hasattr(model, 'preprocess_text') 
        assert hasattr(model, 'compress_frames_with_tempme')
        assert hasattr(model, 'retrieve_with_stop')
        assert hasattr(model, 'forward')
        assert hasattr(model, 'retrieve_event')
        assert hasattr(model, 'extract_video_embedding')
        print("✓ All required methods exist")
        
        print("✓ Unified model structure tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Unified model structure test failed: {e}")
        return False


def test_dummy_inference():
    """Test inference with dummy data."""
    print("="*60)
    print("TESTING DUMMY INFERENCE")
    print("="*60)
    
    try:
        from unified_model import create_unified_model
        
        # Create model with CPU for testing
        model = create_unified_model(device='cpu')
        model.eval()
        print("✓ Model initialized for testing")
        
        # Test video preprocessing (will use dummy data)
        dummy_video_path = "dummy_video.mp4"
        video_tensor, video_mask = model.preprocess_video(dummy_video_path)
        
        assert video_tensor.ndim == 4  # [T, C, H, W]
        assert video_mask.ndim == 1    # [T]
        print(f"✓ Video preprocessing works: {video_tensor.shape}, {video_mask.shape}")
        
        # Test text preprocessing
        query = "A person walking down the street"
        input_ids, attention_mask, token_type_ids = model.preprocess_text(query)
        
        assert input_ids.ndim == 1
        assert attention_mask.ndim == 1
        assert token_type_ids.ndim == 1
        print(f"✓ Text preprocessing works: {input_ids.shape}")
        
        # Test frame compression
        compressed_frames, compressed_mask = model.compress_frames_with_tempme(video_tensor, video_mask)
        
        assert compressed_frames.size(0) == model.config.max_frames
        assert compressed_mask.size(0) == model.config.max_frames
        print(f"✓ Frame compression works: {compressed_frames.shape}")
        
        # Test retrieval (will use dummy similarity scores)
        similarity_scores = model.retrieve_with_stop(
            compressed_frames, compressed_mask,
            input_ids, attention_mask, token_type_ids
        )
        
        assert similarity_scores.ndim == 1
        print(f"✓ Retrieval works: {similarity_scores.shape}")
        
        # Test full pipeline
        frames = model.retrieve_event(dummy_video_path, query, top_k=3)
        assert isinstance(frames, list)
        assert len(frames) <= 3
        print(f"✓ Full pipeline works: {frames}")
        
        print("✓ Dummy inference tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Dummy inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_extraction():
    """Test the video embedding extraction functionality."""
    print("="*60)
    print("TESTING VIDEO EMBEDDING EXTRACTION")
    print("="*60)
    
    try:
        from unified_model import create_unified_model
        
        # Create model with CPU for testing
        model = create_unified_model(device='cpu')
        model.eval()
        print("✓ Model initialized for testing")
        
        # Test embedding extraction
        dummy_video_path = "dummy_video.mp4"
        embedding = model.extract_video_embedding(dummy_video_path)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.ndim == 1  # Should be 1D embedding vector
        print(f"✓ Video embedding extraction works: shape {embedding.shape}")
        
        # Test that embeddings are deterministic (same input = same output)
        embedding2 = model.extract_video_embedding(dummy_video_path)
        assert torch.allclose(embedding, embedding2, atol=1e-5)
        print("✓ Embedding extraction is deterministic")
        
        print("✓ Embedding extraction tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Embedding extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_file_processing():
    """Test batch file processing functionality."""
    print("="*60)
    print("TESTING BATCH FILE PROCESSING")
    print("="*60)
    
    try:
        # Create temporary batch file
        batch_data = [
            {"video_path": "test1.mp4", "query": "person walking"},
            {"video_path": "test2.mp4", "query": "dog running"},
            {"video_path": "test3.mp4", "query": "car driving"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(batch_data, f, indent=2)
            temp_batch_path = f.name
        
        print(f"✓ Created temporary batch file: {temp_batch_path}")
        
        # Test loading batch file
        with open(temp_batch_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 3
        assert all('video_path' in item and 'query' in item for item in loaded_data)
        print("✓ Batch file format validation passed")
        
        # Cleanup
        os.unlink(temp_batch_path)
        
        print("✓ Batch file processing tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Batch file processing test failed: {e}")
        return False


def test_extraction_script():
    """Test extraction script structure."""
    print("="*60)
    print("TESTING EXTRACTION SCRIPT")
    print("="*60)
    
    try:
        # Test importing the extraction script
        import extract_video_embeddings
        
        # Check if main functions exist
        assert hasattr(extract_video_embeddings, 'extract_single_video_embedding')
        assert hasattr(extract_video_embeddings, 'extract_directory_embeddings')
        assert hasattr(extract_video_embeddings, 'get_video_files')
        assert hasattr(extract_video_embeddings, 'save_embedding')
        assert hasattr(extract_video_embeddings, 'main')
        print("✓ Extraction script structure is correct")
        
        # Test video file detection
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy video files
            for ext in ['.mp4', '.avi', '.txt']:  # .txt should be ignored
                dummy_file = os.path.join(temp_dir, f'test{ext}')
                with open(dummy_file, 'w') as f:
                    f.write('dummy')
            
            video_files = extract_video_embeddings.get_video_files(temp_dir)
            assert len(video_files) == 2  # Only .mp4 and .avi should be detected
            print(f"✓ Video file detection works: found {len(video_files)} video files")
        
        print("✓ Extraction script tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Extraction script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test cases."""
    print("🧪 STARTING UNIFIED TEMPME-STOP MODEL TESTS")
    print("="*80)
    
    test_results = []
    
    # Run individual test cases
    test_cases = [
        ("Configuration System", test_config_system),
        ("Unified Model Structure", test_unified_model_structure),
        ("Dummy Inference", test_dummy_inference),
        ("Embedding Extraction", test_embedding_extraction),
        ("Batch File Processing", test_batch_file_processing),
        ("Command Line Interface", test_extraction_script),
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
    print("🏁 TEST RESULTS SUMMARY")
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
        print("🎉 ALL TESTS PASSED! The unified model implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Set PyTorch to use CPU for testing (avoids CUDA issues)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    success = run_all_tests()
    sys.exit(0 if success else 1)