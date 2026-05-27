#!/bin/bash
set -e

# ============ CHANGE DATASET HERE ============
DATASET=toys
# =============================================

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sunwoo

model_path=Qwen/Qwen3.5-4B
output_dir=./model_${DATASET}_sid_sft
special_tokens=$(cat ./data/${DATASET}_special_tokens.txt)

# Aligned codebook embeddings (extracted from alignment checkpoint)
export ALIGNED_EMBED_PATH=./data/${DATASET}_codebook_aligned_embeddings.safetensors
export NUM_NEW_TOKENS=$(echo "$special_tokens" | tr ',' '\n' | grep -c .)

export PYTHONPATH=./src:$PYTHONPATH
export TORCH_DISABLE_ADDR2LINE=1
unset NCCL_NET

torchrun --nproc_per_node=2 \
    train_codebook_weighted_sft.py \
    --deepspeed ds_z2_config.json \
    --stage sft \
    --model_name_or_path $model_path \
    --do_train \
    --dataset_dir train_data \
    --dataset ${DATASET}_sid_merged_sft \
    --template qwen3 \
    --finetuning_type full \
    --add_special_tokens "$special_tokens" \
    --resize_vocab true \
    --output_dir $output_dir \
    --overwrite_cache \
    --save_total_limit 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 700 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.1
