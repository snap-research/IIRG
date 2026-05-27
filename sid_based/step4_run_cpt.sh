#!/bin/bash
set -e

# ============ CHANGE DATASET HERE ============
DATASET=toys
# =============================================
# qwen-3 template works for qwen-3.5 as well, so we keep using it for simplicity

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sunwoo

model_path=Qwen/Qwen3.5-4B
output_dir=./model_${DATASET}_codebook_align
special_tokens=$(cat ./data/${DATASET}_special_tokens.txt)

# Count number of special tokens
NUM_NEW_TOKENS=$(echo "$special_tokens" | tr ',' '\n' | wc -l)
export NUM_NEW_TOKENS
echo "Number of new tokens: $NUM_NEW_TOKENS"

export PYTHONPATH=./src:$PYTHONPATH
export TORCH_DISABLE_ADDR2LINE=1
unset NCCL_NET

torchrun --nproc_per_node=2 \
train_alignment.py \
--deepspeed ds_z2_config.json \
--stage sft \
--model_name_or_path $model_path \
--do_train \
--dataset_dir train_data \
--dataset ${DATASET}_sid_alignment_sft \
--template qwen3 \
--finetuning_type freeze \
--freeze_trainable_layers 0 \
--freeze_extra_modules embed_tokens \
--add_special_tokens "$special_tokens" \
--resize_vocab true \
--output_dir $output_dir \
--overwrite_cache \
--save_total_limit 1 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 500 \
--learning_rate 1e-3 \
--num_train_epochs 10.0 \
--plot_loss \
--bf16 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--warmup_ratio 0.05
