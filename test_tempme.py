#!/usr/bin/env python3
"""
Test script for TempMe integration with STOP model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
from modules.tempme import SimpleTempMe, get_tempme_model


def test_simple_tempme():
    """Test SimpleTempMe with different input sizes"""
    print("Testing SimpleTempMe...")
    
    # Create model
    tempme = SimpleTempMe(compression_ratio=0.75, hidden_dim=256)
    
    # Test with different video sizes
    test_cases = [
        (2, 12, 3, 224, 224),  # Standard STOP input
        (1, 16, 3, 224, 224),  # Slightly more frames
        (2, 8, 3, 224, 224),   # Fewer frames
    ]
    
    for batch_size, frames, channels, height, width in test_cases:
        print(f"Testing input shape: [{batch_size}, {frames}, {channels}, {height}, {width}]")
        
        # Create random input
        video_input = torch.randn(batch_size, frames, channels, height, width)
        
        # Apply TempMe
        with torch.no_grad():
            compressed = tempme(video_input)
        
        print(f"  Input frames: {frames}")
        print(f"  Output frames: {compressed.size(1)}")
        print(f"  Compression ratio: {compressed.size(1) / frames:.2f}")
        print(f"  Output shape: {list(compressed.shape)}")
        print()
    
    print("SimpleTempMe test completed successfully!")


def test_tempme_factory():
    """Test TempMe factory function"""
    print("Testing TempMe factory function...")
    
    # Create mock config
    class MockConfig:
        def __init__(self):
            self.tempme_type = 'simple'
            self.tempme_compression_ratio = 0.6
            self.tempme_hidden_dim = 128
    
    config = MockConfig()
    
    # Test factory function
    tempme = get_tempme_model(config)
    
    # Test with sample input
    video_input = torch.randn(1, 12, 3, 224, 224)
    
    with torch.no_grad():
        compressed = tempme(video_input)
    
    print(f"Factory test - Input: {list(video_input.shape)}")
    print(f"Factory test - Output: {list(compressed.shape)}")
    print("TempMe factory test completed successfully!")


def test_integration_ready():
    """Test that TempMe is ready for integration"""
    print("Testing integration readiness...")
    
    # Test with minimal setup similar to CLIP4Clip
    try:
        from modules.tempme import SimpleTempMe
        from modules.clip4clip import CLIP4Clip
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test TempMe initialization
    try:
        tempme = SimpleTempMe()
        print("✓ TempMe initialization successful")
    except Exception as e:
        print(f"✗ TempMe initialization failed: {e}")
        return False
    
    # Test forward pass
    try:
        video_input = torch.randn(2, 12, 3, 224, 224)
        with torch.no_grad():
            output = tempme(video_input)
        print(f"✓ Forward pass successful: {list(output.shape)}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    print("Integration readiness test completed successfully!")
    return True


if __name__ == "__main__":
    print("TempMe Integration Test")
    print("=" * 50)
    
    # Run tests
    test_simple_tempme()
    print()
    test_tempme_factory()
    print()
    test_integration_ready()
    
    print("\nAll tests completed!")