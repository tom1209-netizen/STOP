# Unified STOP-TempMe Model Implementation

This document describes the implementation of the unified model that integrates TempMe and STOP into a single end-to-end pipeline for temporal video event retrieval.

## Overview

The unified model combines:
- **TempMe approach**: Frame compression to reduce temporal redundancy
- **STOP model**: Video-text retrieval using CLIP backbone

### Pipeline Flow
1. **Input**: Raw video with variable number of frames (e.g., 24, 50, 100 frames)
2. **Frame Compression**: TempMe-inspired compression to exactly 12 representative frames
3. **STOP Processing**: The 12 frames are processed by STOP model for video-text retrieval
4. **Output**: Video embeddings for similarity computation with text queries

## Implementation Details

### File Structure
```
/modules/                     # Symlink to stop-modules (original STOP implementation)
/unified_models/              # New unified model implementation
├── __init__.py
└── unified_stop_tempme.py    # Main unified model class
/params.py                    # Updated with --model_type parameter
/main.py                      # Updated to support model selection
```

### Key Components

#### 1. UnifiedStopTempMe Class
Located in `unified_models/unified_stop_tempme.py`

**Key Methods:**
- `compress_video_frames(video, video_mask)`: Compresses video to 12 frames
- `forward(...)`: End-to-end pipeline with text and video inputs
- `from_pretrained(...)`: Creates model from pretrained STOP components

#### 2. Frame Compression Algorithm
Currently implements **uniform sampling** for frame compression:
- Videos with >12 frames: Sample 12 evenly spaced frames
- Videos with =12 frames: Pass through unchanged  
- Videos with <12 frames: Pad with zeros to reach 12 frames

*Future enhancement: Replace with TempMe's token merging approach*

#### 3. Model Selection
New parameter in `params.py`:
```bash
--model_type {stop,unified}  # Choose between original STOP or unified model
```

## Usage

### Training with Unified Model
```bash
python main.py \
    --model_type unified \
    --output_dir /path/to/output \
    --batch_size 32 \
    --max_frames 24 \
    --epochs 5 \
    --datatype msrvtt
```

### Training with Original STOP Model  
```bash
python main.py \
    --model_type stop \
    --output_dir /path/to/output \
    --batch_size 32 \
    --max_frames 12 \
    --epochs 5 \
    --datatype msrvtt
```

## Testing

### Run All Tests
```bash
# Test frame compression pipeline
python test_frame_compression.py

# Test model integration 
python test_integration.py

# Test unified model functionality
python test_unified_model.py
```

### Test Results
✅ All tests passing:
- Frame compression handles videos of any length
- Unified model integrates properly with training pipeline
- Model selection logic works correctly
- Compatible with existing STOP training procedures

## Benefits

1. **Handles Variable-Length Videos**: Can process videos with any number of frames
2. **Efficient Processing**: Compresses long videos to fixed 12-frame representation
3. **Backward Compatible**: Original STOP model still available via `--model_type stop`
4. **Easy Integration**: Drop-in replacement in existing training pipeline

## Future Enhancements

1. **Enhanced Frame Compression**: 
   - Replace uniform sampling with TempMe's token merging
   - Add learnable frame selection
   
2. **Dynamic Frame Count**:
   - Support variable target frame counts
   - Adaptive compression based on video content
   
3. **Full TempMe Integration**:
   - Complete integration of TempMe VTRModel
   - Token-level merging in addition to frame-level compression

## Architecture Diagram

```
Input Video (T frames)
        ↓
[Frame Compression]  ← TempMe-inspired
        ↓
12 Representative Frames
        ↓
[STOP Model]        ← Original STOP/CLIP4Clip
        ↓
Video Embeddings
```

## Validation

The implementation has been validated with:
- ✅ Multiple frame count scenarios (1, 8, 12, 24, 100 frames)
- ✅ Batch processing
- ✅ Forward pass compatibility
- ✅ Training pipeline integration
- ✅ Model selection functionality

The unified model is ready for production use and training on video-text retrieval datasets.