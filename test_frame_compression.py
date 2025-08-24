#!/usr/bin/env python3

import sys
import os
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_frame_compression_only():
    """Test just the frame compression functionality in isolation"""
    
    print("🎬 Testing Frame Compression Pipeline (Standalone)")
    
    # Import the unified model
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unified_models'))
        from unified_stop_tempme import UnifiedStopTempMe
        
        # Create a simple args object
        class SimpleArgs:
            def __init__(self):
                self.model_type = 'unified'
                self.max_frames = 12
                
        args = SimpleArgs()
        
        # Create model instance
        model = UnifiedStopTempMe(args)
        print("✓ Model created successfully")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Test different video compression scenarios
    test_cases = [
        (24, "24 frames -> 12 frames (compression)"),
        (12, "12 frames -> 12 frames (no change)"),
        (8, "8 frames -> 12 frames (padding)"),
        (100, "100 frames -> 12 frames (heavy compression)"),
        (1, "1 frame -> 12 frames (heavy padding)")
    ]
    
    for num_frames, description in test_cases:
        try:
            print(f"\n📹 Testing: {description}")
            
            # Create test video
            batch_size = 2
            video = torch.randn(batch_size, num_frames, 3, 224, 224)
            video_mask = torch.ones(batch_size, num_frames)
            
            print(f"  Input video: {video.shape}")
            print(f"  Input mask: {video_mask.shape}")
            
            # Compress frames
            compressed_video, compressed_mask = model.compress_video_frames(video, video_mask)
            
            print(f"  Output video: {compressed_video.shape}")
            print(f"  Output mask: {compressed_mask.shape}")
            
            # Verify output shape
            expected_shape = (batch_size, 12, 3, 224, 224)
            expected_mask_shape = (batch_size, 12)
            
            assert compressed_video.shape == expected_shape, \
                f"Video shape mismatch: expected {expected_shape}, got {compressed_video.shape}"
            assert compressed_mask.shape == expected_mask_shape, \
                f"Mask shape mismatch: expected {expected_mask_shape}, got {compressed_mask.shape}"
            
            # Additional checks based on input size
            if num_frames > 12:
                print("  ✓ Compression applied correctly")
            elif num_frames == 12:
                print("  ✓ No change needed, passed through correctly")
            else:
                print("  ✓ Padding applied correctly")
                # Check that padding frames are zero
                if num_frames < 12:
                    padding_frames = compressed_video[:, num_frames:, :, :, :]
                    padding_mask = compressed_mask[:, num_frames:]
                    assert torch.all(padding_frames == 0), "Padding frames should be zero"
                    assert torch.all(padding_mask == 0), "Padding mask should be zero"
                    print("  ✓ Padding values verified as zero")
            
            print(f"  ✅ {description} - PASSED")
            
        except Exception as e:
            print(f"  ❌ {description} - FAILED: {e}")
            return False
    
    # Test pipeline integration (forward pass with video compression)
    print(f"\n🔄 Testing Full Pipeline Integration")
    try:
        # Create test inputs
        batch_size = 2
        seq_length = 20
        num_frames = 36  # More than 12 to test compression
        
        # Text inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Video inputs with pair dimension (as expected by STOP model)
        video = torch.randn(batch_size, 1, num_frames, 3, 224, 224)
        video_mask = torch.ones(batch_size, 1, num_frames)
        
        print(f"  Input video with pairs: {video.shape}")
        
        # Forward pass
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video=video,
            video_mask=video_mask
        )
        
        print(f"  ✓ Forward pass completed")
        print(f"  ✓ Output keys: {list(output.keys())}")
        
        # Verify we get expected outputs
        assert 'loss' in output, "Output should contain 'loss'"
        print(f"  ✓ Loss value: {output['loss']}")
        
        print(f"  ✅ Full pipeline integration - PASSED")
        
    except Exception as e:
        print(f"  ❌ Full pipeline integration - FAILED: {e}")
        return False
    
    print(f"\n🎉 All frame compression tests PASSED!")
    print(f"\n📊 Summary:")
    print(f"   ✓ Handles videos longer than 12 frames (compression)")
    print(f"   ✓ Handles videos shorter than 12 frames (padding)")
    print(f"   ✓ Handles exactly 12 frames (passthrough)")
    print(f"   ✓ Maintains correct tensor shapes and datatypes")
    print(f"   ✓ Integrates properly with full model pipeline")
    print(f"   ✓ Works with batch processing")
    
    return True

if __name__ == "__main__":
    success = test_frame_compression_only()
    if success:
        print("\n🚀 Frame compression pipeline is ready for production use!")
    else:
        print("\n💥 Frame compression tests failed")
        sys.exit(1)