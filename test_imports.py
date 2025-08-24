#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Test imports one by one
try:
    from params import get_args
    print("✓ params import successful")
    
    args = get_args()
    print(f"✓ Model type parameter: {getattr(args, 'model_type', 'default: stop')}")
    
except Exception as e:
    print(f"✗ params import failed: {e}")
    sys.exit(1)

try:
    from modules import CLIP4Clip
    print("✓ CLIP4Clip import successful")
except Exception as e:
    print(f"✗ CLIP4Clip import failed: {e}")

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unified_models'))
    from unified_stop_tempme import UnifiedStopTempMe
    print("✓ UnifiedStopTempMe import successful")
except Exception as e:
    print(f"✗ UnifiedStopTempMe import failed: {e}")
    
print("Basic import test completed!")