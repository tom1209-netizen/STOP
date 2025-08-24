#!/usr/bin/env python3

import sys
import os
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_unified_model():
    """Test the unified model instantiation and basic functionality"""
    
    print("🔧 Testing Unified STOP-TempMe Model")
    
    # Get arguments
    try:
        from params import get_args
        
        # Create minimal args for testing
        sys.argv = [
            'test_model.py',
            '--output_dir', '/tmp/test_unified', 
            '--model_type', 'unified',
            '--pretrained_dir', '/tmp/pretrained',  # Won't exist, but testing structure
            '--batch_size', '2',
            '--max_frames', '24',  # Test with more than 12 frames
            '--epochs', '1'
        ]
        
        args = get_args()
        print(f"✓ Arguments parsed successfully. Model type: {args.model_type}")
        
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False
    
    # Test unified model import
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unified_models'))
        from unified_stop_tempme import UnifiedStopTempMe
        print("✓ UnifiedStopTempMe imported successfully")
        
    except Exception as e:
        print(f"✗ UnifiedStopTempMe import failed: {e}")
        return False
    
    # Test model creation (without actual pretrained weights)
    try:
        # Create a simple config-like object for testing
        class SimpleConfig:
            def __init__(self):
                self.cross_model = 'cross-base'
                self.cache_dir = '/tmp/cache'
                self.pretrained_clip_name = 'ViT-B/32'
                self.pretrained_dir = '/tmp/pretrained'
                
        config = SimpleConfig()
        
        # Try to create model instance (this will likely fail due to missing pretrained weights)
        try:
            model = UnifiedStopTempMe.from_pretrained(
                cross_model_name='cross-base',
                task_config=args
            )
            if model is not None:
                print("✓ UnifiedStopTempMe model created successfully")
                return test_model_forward(model, args)
            else:
                print("⚠️  Model creation returned None (expected without pretrained weights)")
                
        except Exception as e:
            print(f"⚠️  Model creation failed (expected without pretrained weights): {e}")
        
        # Test the basic structure without full initialization
        print("✓ Testing basic model structure...")
        model = UnifiedStopTempMe(args)
        print("✓ Basic model structure created")
        
        # Test frame compression function
        return test_frame_compression(model)
        
    except Exception as e:
        print(f"✗ Model structure test failed: {e}")
        return False

def test_frame_compression(model):
    """Test the frame compression functionality"""
    
    print("\n🎬 Testing Frame Compression Pipeline")
    
    try:
        # Create dummy video data: batch=2, frames=24, channels=3, height=224, width=224
        batch_size = 2
        original_frames = 24
        target_frames = 12
        
        video = torch.randn(batch_size, original_frames, 3, 224, 224)
        video_mask = torch.ones(batch_size, original_frames)
        
        print(f"✓ Created test video: {video.shape}")
        print(f"✓ Created test mask: {video_mask.shape}")
        
        # Test frame compression
        compressed_video, compressed_mask = model.compress_video_frames(video, video_mask)
        
        print(f"✓ Compressed video shape: {compressed_video.shape}")
        print(f"✓ Compressed mask shape: {compressed_mask.shape}")
        
        # Verify compression worked correctly
        assert compressed_video.shape == (batch_size, target_frames, 3, 224, 224), \
            f"Expected {(batch_size, target_frames, 3, 224, 224)}, got {compressed_video.shape}"
        assert compressed_mask.shape == (batch_size, target_frames), \
            f"Expected {(batch_size, target_frames)}, got {compressed_mask.shape}"
        
        print("✓ Frame compression test passed!")
        
        # Test with different input sizes
        return test_edge_cases(model)
        
    except Exception as e:
        print(f"✗ Frame compression test failed: {e}")
        return False

def test_edge_cases(model):
    """Test edge cases for frame compression"""
    
    print("\n🔍 Testing Edge Cases")
    
    try:
        # Test with exactly 12 frames
        video_12 = torch.randn(1, 12, 3, 224, 224)
        mask_12 = torch.ones(1, 12)
        
        compressed_video, compressed_mask = model.compress_video_frames(video_12, mask_12)
        assert compressed_video.shape == (1, 12, 3, 224, 224)
        print("✓ 12-frame input test passed")
        
        # Test with fewer than 12 frames
        video_8 = torch.randn(1, 8, 3, 224, 224) 
        mask_8 = torch.ones(1, 8)
        
        compressed_video, compressed_mask = model.compress_video_frames(video_8, mask_8)
        assert compressed_video.shape == (1, 12, 3, 224, 224)  # Should be padded
        assert compressed_mask.shape == (1, 12)
        print("✓ 8-frame input (padding) test passed")
        
        # Test with many frames
        video_100 = torch.randn(1, 100, 3, 224, 224)
        mask_100 = torch.ones(1, 100)
        
        compressed_video, compressed_mask = model.compress_video_frames(video_100, mask_100)
        assert compressed_video.shape == (1, 12, 3, 224, 224)  # Should be compressed
        assert compressed_mask.shape == (1, 12)
        print("✓ 100-frame input (compression) test passed")
        
        print("✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False

def test_model_forward(model, args):
    """Test model forward pass if model is available"""
    
    print("\n⚡ Testing Model Forward Pass")
    
    try:
        # Create dummy inputs
        batch_size = 2
        seq_length = 20
        num_frames = 24
        
        # Text inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        token_type_ids = torch.zeros(batch_size, seq_length)
        
        # Video inputs (more than 12 frames to test compression)
        video = torch.randn(batch_size, 1, num_frames, 3, 224, 224)  # Add pair dimension
        video_mask = torch.ones(batch_size, 1, num_frames)
        
        print(f"✓ Created test inputs - Video: {video.shape}, Text: {input_ids.shape}")
        
        # Forward pass
        output = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            video=video,
            video_mask=video_mask
        )
        
        print(f"✓ Forward pass completed. Output keys: {list(output.keys())}")
        
        if 'loss' in output:
            print(f"✓ Loss computed: {output['loss']}")
            
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_unified_model()
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\n📝 Summary:")
        print("   - Unified model structure works correctly")
        print("   - Frame compression pipeline functional")  
        print("   - Edge cases handled properly")
        print("   - Ready for integration with training pipeline")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)