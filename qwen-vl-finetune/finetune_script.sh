#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# Path Configuration
# ======================
# MODEL_PATH="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct-AWQ/"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="./checkpoints"
CACHE_DIR="./cache"

# ======================
# Model Configuration
# ======================
DATASETS="my_dataset%100"

# ======================
# Compute pixel counts
# ======================
MAX_PIXELS=$((576 * 28 * 28))
MIN_PIXELS=$((16 * 28 * 28))
VIDEO_MAX_FRAME_PIXELS=$((1664 * 28 * 28))
VIDEO_MIN_FRAME_PIXELS=$((256 * 28 * 28))

# ======================
# Launch Training
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm False \
         --tune_mm_vision False \
         --tune_mm_mlp True \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --data_packing True \
         --max_pixels $MAX_PIXELS \
         --min_pixels $MIN_PIXELS \
         --base_interval 2 \
         --video_max_frames 8 \
         --video_min_frames 4 \
         --video_max_frame_pixels $VIDEO_MAX_FRAME_PIXELS \
         --video_min_frame_pixels $VIDEO_MIN_FRAME_PIXELS \
         --num_train_epochs 3 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 500 \
         --save_total_limit 3 \
         --deepspeed zero3.json

echo "====================================================="
echo "Reminder: Add \"_attn_implementation\": \"flash_attention_2\""
echo "to your model's config.json if you want to enable FlashAttention."
echo "====================================================="

