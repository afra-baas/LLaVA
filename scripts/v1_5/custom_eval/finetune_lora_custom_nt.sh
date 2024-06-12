#!/bin/bash

# WANDB_MODE=offline
epochs=$1
version=$2
dataset=$3
method=$4
device=$5

# method="no_depth"
# method="conv"
# method="conv_hyper_param_lr5_nwd"
# method="late"
# method="mlp"
# method="method2" # dont forget to change init in model llava_llama

if [ "$version" = "7b" ]; then
    model_base="liuhaotian/llava-v1.5-7b"
    model="llava-v1.5-7b"
elif [ "$version" = "13b" ]; then
    model_base="liuhaotian/llava-v1.5-13b"
    model="llava-v1.5-13b"
fi

############################################# VSR ######################################
if [ "$dataset" = "VSR" ]; then
    train_file="VSR_train_TF.json"
    val_file="VSR_val_TF.json"

    # train_file="VSR_train_TF_copy.json"
    # val_file="VSR_val_TF_copy.json"

    CKPT="VSR_TF_epoch${epochs}-${method}-${model}"

    depth_train_file="/project/msc-thesis-project/vsr_depth/train/"
    depth_val_file="/project/msc-thesis-project/vsr_depth/val/"

elif [ "$dataset" = "VSR_class" ]; then
    train_file="VSR_train_classification.json"
    val_file="VSR_val_classification.json"

    CKPT="VSR_epoch${epochs}-${method}-${model}"

    depth_train_file="/project/msc-thesis-project/vsr_depth/train/"
    depth_val_file="/project/msc-thesis-project/vsr_depth/val/"

elif [ "$dataset" = "VSR_midas" ]; then
    train_file="VSR_train_TF_new.json"
    val_file="VSR_val_TF_new.json"

    CKPT="VSR_TF_midas_epoch${epochs}-${method}-${model}"

    # TODO: change gd to midas in all file names
    depth_train_file="/project/msc-thesis-project/vsr_depth_gd/train/"
    depth_val_file="/project/msc-thesis-project/vsr_depth_gd/val/"

############################################ What UP #######################################
elif [ "$dataset" = "Whatsup" ]; then
    train_file="whatsup_train_classification_controlled_images.json"
    val_file="whatsup_val_classification_controlled_images.json"

    CKPT="epoch${epochs}-${method}-${model}"

    depth_train_file="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"
    depth_val_file="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"

####################################################################################

# train_file="whatsup_train_classification_COCO.json"
# fusion_method="train.py"
# CKPT="WhatsupCOCO_epoch3-nodepth-llava-v1.5-7b"
# # CKPT="WhatsupCOCO_epoch3-nodepth-llava-v1.5-13b"

########################################## Different Model ###############################

# train_file="VSR_train_TF_copy.json"
# val_file="VSR_val_TF_copy.json"

# # train_file="whatsup_train_classification_controlled_images.json"
# # val_file="whatsup_val_classification_controlled_images.json"

# fusion_method="train.py"
# model_base="Salesforce/instructblip-vicuna-7b"
# CKPT="VSR_TF_epoch3-nodepth-instructionblip_testing"

###########################################################################################
    # --deepspeed ./scripts/zero3.json \

fi # end of ifs

if [ "$method" = "no_depth" ]; then
    fusion_method="train.py"
elif [ "$method" = "conv" ]; then
    fusion_method="train_custom.py"
elif [ "$method" = "mlp" ]; then
    fusion_method="train_custom_mlp.py"
elif [ "$method" = "method2" ]; then
    fusion_method="train_custom_method2.py"
elif [ "$method" = "late" ]; then
    fusion_method="train_custom_late.py"
elif [ "$method" = "resnet" ]; then
    fusion_method="train_custom_resnet.py"
elif [ "$method" = "resnet_fusion" ]; then
    fusion_method="train_custom_resnet.py"
elif [ "$method" = "de" ]; then
    fusion_method="train_custom_de.py"
elif [ "$method" = "dino" ]; then
    fusion_method="train_custom_dino.py"
else
    echo "Invalid method specified."
    # fusion_method="train_custom.py"
fi


# --mm_use_im_patch_token False \
lr=2e-4
# lr=2e-5
echo "Finetuning $model_base with $fusion_method on $train_file for $epochs epochs to become $CKPT"
# CUDA_VISIBLE_DEVICES=0 deepspeed /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method \
CUDA_VISIBLE_DEVICES=$device python /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $model_base \
    --version v1 \
    --data_path /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${train_file} \
    --validation_data_path /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${val_file} \
    --depth_path_train ${depth_train_file} \
    --depth_path_val ${depth_val_file} \
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
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $lr \
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
