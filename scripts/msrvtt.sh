#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Suppress TensorFlow and CUDA verbose messages
export TF_CPP_MIN_LOG_LEVEL=3               # Suppress TensorFlow info/warning/error messages
export TF_ENABLE_ONEDNN_OPTS=0              # Disable oneDNN custom operations
export CUDA_LAUNCH_BLOCKING=0               # Disable CUDA launch blocking
export PYTHONWARNINGS="ignore"              # Suppress Python warnings
export TOKENIZERS_PARALLELISM=false         # Suppress tokenizers parallelism warnings

echo "Using local machine for training"



group=group2-2


# dataset
dataset=msrvtt
fps=3

DATA_PATH=/content/MSRVTT
data_path=${DATA_PATH}/annotation/MSRVTT_data.json
train_csv=${DATA_PATH}/annotation/MSRVTT_train.9k.csv

# training-9k
val_csv=${DATA_PATH}/annotation/MSRVTT_JSFUSION_test.csv
features_path=${DATA_PATH}/compressed_videos/
pretrained_dir=/content/STOP/modules

# train or eval
do_train=1
do_eval=0


# learning strategies
pretrained_clip_name=ViT-B/32
lr=1e-3
coef_lr=5e-4
wd=0.2
epochs=5
optim=AdamW
max_words=32
max_frames=12
temperature_new=1.0
resume=None
load_from_pretrained=0
batch_size=32           # single GPU batch size
batch_size_val=16
num_workers=8
n_display=50            # log per n_display
precision=amp

freeze_clip=1
time_embedding=0

shared_latent_space=transformer

# ToMe (Token Merging) parameters - from TempMe integration
# Expected benefits: 30-60% memory reduction, 25-50% speed improvement
tome_r=2                          # Number of tokens to merge per layer (0 to disable)
tome_trace_source=false           # Whether to trace source tokens for debugging
tome_prop_attn=true              # Whether to propagate attention with size information

# Inter-frame merging configuration
merge_layers="3-6-9"             # Layers where inter-frame merging occurs
merge_frame_nums="2-2-2"         # Frame merge ratios for each merge layer
merge_token_proportions="10-10"  # Token merge proportions: inter_frame%-intra_frame%
frame_pos=0                      # Whether to use frame positional embeddings (0=no, 1=yes)

# Alternative ToMe configurations (uncomment to use):
# Conservative (high accuracy): tome_r=1, merge_layers="6-9", merge_frame_nums="2-2", merge_token_proportions="5-5"
# Aggressive (max speed): tome_r=4, merge_layers="2-4-6-8-10", merge_frame_nums="2-2-3-3-3", merge_token_proportions="15-15"
# Disable ToMe: tome_r=0

# distributed training
# init_method='tcp://127.0.0.1:6010'




current_datetime=$(TZ="Asia/Tokyo" date +"%Y-%m-%d-%H:%M:%S")
model_dir=/content/logs/${current_datetime}_${dataset}_STOP
echo "The model dir is ${model_dir}"

# Redirect stderr to filter out CUDA/TensorFlow verbose messages
python main.py \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --num_thread_reader ${num_workers} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --n_display ${n_display} \
        --train_csv ${train_csv} \
        --val_csv ${val_csv} \
        --data_path ${data_path} \
        --features_path ${features_path} \
        --output_dir ${model_dir} \
        --optim ${optim} \
        --lr ${lr} \
        --coef_lr ${coef_lr} \
        --wd ${wd} \
        --max_words ${max_words} \
        --max_frames ${max_frames} \
        --batch_size_val ${batch_size_val} \
        --datatype ${dataset} \
        --expand_msrvtt_sentences  \
        --feature_framerate ${fps} \
        --freeze_layer_num 12  \
        --slice_framepos 2 \
        --loose_type \
        --linear_patch 2d \
        --sim_header meanP \
        --pretrained_clip_name ${pretrained_clip_name} \
        --precision ${precision} \
        --pretrained_dir ${pretrained_dir} \
        --freeze_clip ${freeze_clip} \
        --time_embedding ${time_embedding} \
        --resume ${resume} \
        --load_from_pretrained ${load_from_pretrained} \
        --shared_latent_space ${shared_latent_space} \
        --temporal_prompt ${group} \
        --tome_r ${tome_r} \
        --tome_trace_source \
        --tome_prop_attn \
        --merge_layers ${merge_layers} \
        --merge_frame_nums ${merge_frame_nums} \
        --merge_token_proportions ${merge_token_proportions} \
        --frame_pos ${frame_pos} 2>&1 | grep -v -E "(Unable to register|computation placer already registered|This TensorFlow binary is optimized|oneDNN custom operations|All log messages before absl::InitializeLog)"


echo "Training Finished!!!"
