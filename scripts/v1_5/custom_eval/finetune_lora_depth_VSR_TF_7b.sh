#!/bin/bash
# llava/train/train_mem.py \

# WANDB_MODE=offline
train_file="VSR_train_TF_copy.json"
val_file="VSR_val_TF_copy.json"
epochs=1
CKPT="VSR_TF_epoch3-depth-llava-v1.5-7b_testing"

deepspeed /project/msc-thesis-project/forked_repos/LLaVA/llava/train/train_custom.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${train_file} \
    --validation_data_path /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${val_file} \
    --image_folder ''\
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/checkpoint-${CKPT}-lora \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb\
    --run_name $CKPT
