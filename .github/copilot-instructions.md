# STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- **CRITICAL: NEVER CANCEL environment setup** - Environment creation takes 15-20 minutes. Set timeout to 30+ minutes.
- Create conda environment: `conda env create -f environment.yaml` -- takes 15-20 minutes. NEVER CANCEL.
- Activate environment: `source /usr/share/miniconda/etc/profile.d/conda.sh && conda activate cuda118`
- Verify setup: `python -c "import torch; from modules import CLIP4Clip; print('Setup successful')"`

### Download Required Models
- **CRITICAL: Download takes 5-10 minutes** - Model download can be slow due to file size (150MB+). Set timeout to 15+ minutes.
- Create model directory: `mkdir -p ${HOME}/models/pretrained`
- Download CLIP model: `wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -P ${HOME}/models/pretrained` -- takes 5-10 minutes. NEVER CANCEL.
- Update `pretrained_dir` in training scripts to point to your model directory

### Dataset Preparation  
- **CRITICAL: Video preprocessing is time-intensive** - Each video takes 1-5 seconds to process. Large datasets take hours. Set timeout to 120+ minutes.
- Download datasets (MSRVTT, ActivityNet, VATEX) using links in readme.md
- Preprocess videos: `python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]` -- takes 1-3 hours for full dataset. NEVER CANCEL.
- Videos are compressed to 3fps with 224px resolution for faster training

### Training
- **CRITICAL: Training takes 2-6 hours per epoch** - Full training for 5 epochs takes 10-30 hours. Set timeout to 1800+ minutes (30+ hours).
- MSRVTT dataset: `chmod +x ./scripts/msrvtt.sh && bash ./scripts/msrvtt.sh` -- takes 10-30 hours for full training. NEVER CANCEL.
- ActivityNet dataset: `chmod +x ./scripts/activitynet.sh && bash ./scripts/activitynet.sh` -- takes 15-40 hours for full training. NEVER CANCEL.
- VATEX dataset: `chmod +x ./scripts/vatex.sh && bash ./scripts/vatex.sh` -- takes 20-50 hours for full training. NEVER CANCEL.
- Training logs saved to timestamped directories (e.g., `logs/2024-01-01-12:00:00_msrvtt_STOP`)

### Key Parameters to Modify
Always update these paths in training scripts before running:
- `DATA_PATH`: Point to your dataset root directory
- `pretrained_dir`: Point to directory containing ViT-B-32.pt
- `features_path`: Point to compressed video directory
- `CUDA_VISIBLE_DEVICES`: Set GPU devices (e.g., "0,1,2,3" for 4 GPUs)

## Validation

### Environment Validation
- Always run after environment setup: `python -c "import torch, torchvision, transformers; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"`
- Test module imports: `python -c "from modules import CLIP4Clip; from dataloaders.data_dataloaders import DATALOADER_DICT; print('All modules imported successfully')"`
- Verify ffmpeg for video processing: `ffmpeg -version`

### Training Validation
- **ALWAYS test training scripts before full runs** to catch configuration errors early
- Test with minimal settings: Modify script to use `epochs=1`, `batch_size=2`, small dataset subset
- Quick validation run: `python main.py --help` to verify all parameters are recognized
- **MANUAL VALIDATION REQUIREMENT**: After successful training, manually verify:
  - Model checkpoints are saved in output directory
  - Training logs show decreasing loss
  - Tensorboard logs are created (if enabled)
  - Video-text retrieval metrics improve over epochs

### Troubleshooting Common Issues
- **CUDA out of memory**: Reduce `batch_size` in training scripts (try 8, 4, 2, 1)
- **Model loading failed**: Verify `ViT-B-32.pt` exists in `pretrained_dir` and is not corrupted
- **Video processing failed**: Check ffmpeg installation and video file permissions
- **Dataset loading failed**: Verify CSV files and video paths match your directory structure

## Common Tasks

### Quick Start (for development)
1. `conda env create -f environment.yaml` -- 15-20 min
2. `conda activate cuda118`
3. Download CLIP model to `${HOME}/models/pretrained/` -- 5-10 min  
4. Modify script paths in `scripts/msrvtt.sh`
5. `bash scripts/msrvtt.sh` -- 10-30 hours

### Testing Changes
- Always run parameter validation: `python main.py --output_dir /tmp/test --do_train 0 --do_eval 1 --pretrained_dir ${HOME}/models/pretrained --pretrained_clip_name ViT-B/32 --datatype msrvtt --val_csv /path/to/val.csv --data_path /path/to/data.json --features_path /path/to/videos`
- Quick smoke test: Use 1 epoch, small batch size, subset of data
- Full validation requires running complete training pipeline

### Repository Structure Reference
```
stop-implementation/
├── main.py                 # Main training script
├── params.py               # Parameter definitions  
├── modules/                # Core model implementations
│   ├── clip4clip.py       # Main CLIP4Clip model
│   ├── temporal_prompting.py # STOP temporal prompting
│   └── ...
├── scripts/                # Training scripts for different datasets
│   ├── msrvtt.sh          # MSRVTT training (5 epochs, ~10-30 hours)
│   ├── activitynet.sh     # ActivityNet training (10 epochs, ~15-40 hours)  
│   └── vatex.sh           # VATEX training
├── dataloaders/            # Dataset loading utilities
├── preprocess/             # Video preprocessing scripts
└── utils/                  # Training utilities
```

### Expected Timing Reference
- Environment setup: 15-20 minutes
- CLIP model download: 5-10 minutes  
- Video preprocessing: 1-3 hours (full dataset)
- Model initialization: 2-5 seconds
- Training epoch (MSRVTT): 2-6 hours
- Training epoch (ActivityNet): 3-8 hours
- Full training pipeline: 10-50 hours depending on dataset

### Hardware Requirements
- GPU: Recommended 8GB+ VRAM (can run on CPU but extremely slow)
- RAM: 16GB+ recommended for dataset loading
- Storage: 50GB+ for datasets and model checkpoints
- Network: Stable connection for model/dataset downloads

## Critical Reminders
- **NEVER CANCEL long-running operations** - Use appropriate timeouts (30+ minutes for setup, 30+ hours for training)
- **Always test with small configurations first** before running full training
- **Manually validate functionality** after training completion by checking outputs and logs
- **Update all file paths** in training scripts to match your environment
- **Monitor training progress** through logs and tensorboard (if enabled)