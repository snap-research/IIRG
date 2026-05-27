#!/bin/bash
set -e

# ============ CHANGE DATASET HERE ============
DATASET=toys
# =============================================

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sunwoo

model_path=Qwen/Qwen3.5-4B
output_dir=./model_${DATASET}_tid_sft

export PYTHONPATH=./src:$PYTHONPATH
export TORCH_DISABLE_ADDR2LINE=1
unset NCCL_NET

torchrun --nproc_per_node=2 \
    train_weighted_sft.py \
    --deepspeed ds_z2_config.json \
    --stage sft \
    --model_name_or_path $model_path \
    --do_train \
    --dataset_dir train_data \
    --dataset ${DATASET}_tid_merged_sft \
    --template qwen3 \
    --finetuning_type full \
    --output_dir $output_dir \
    --overwrite_cache \
    --save_total_limit 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 700 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.1
