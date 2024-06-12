#!/bin/bash
# llava/train/train_mem.py \
# only one of VSR or Whatsup on at the time

echo $WANDB_API_KEY

# WANDB_MODE=offline
# lr=$1
# weight_decay=$2
# warmup_ratio=$3
# lr_scheduler_type=$4
# mm_vision_select_layer=$5
# mm_projector_lr=$6
# momentum=$7
# dropout_rate=$8
# dataset=$9
# method_=${10}
# train_batchsize=${11}
# eval_batchsize=${12}

mm_projector_lr=$1
lr=$2
# lr_scheduler_type=$3
dataset="VSR"
method_="dino"
train_batchsize=16
eval_batchsize=16

epochs=3
version="7b"
device=0
weight_decay=0.0001
# lr=2e-4
warmup_ratio=0.03
lr_scheduler_type="cosine"
mm_vision_select_layer=-2



echo "learning_rate: $lr"
echo "weight_decay: $weight_decay"
echo "warmup_ratio: $warmup_ratio"
echo "lr_scheduler_type: $lr_scheduler_type"
echo "mm_vision_select_layer: $mm_vision_select_layer"
echo "mm_projector_lr: $mm_projector_lr"
echo "momentum: $momentum"
echo "dropout_rate: $dropout_rate"
echo "dataset: $dataset"
echo "method: $method_"
echo "train_batchsize: $train_batchsize"
echo "eval_batchsize: $eval_batchsize"



# dataset and method_ (you can change this based on your requirement)
# dataset="VSR"
# dataset="VSR_class2"
# dataset="VSR_combi"
# dataset="VSR_class"
# dataset="Whatsup"
# dataset="VSR_random"
# dataset="VSR_random_BCE"
# dataset="VSR_random_class"


# method_="no_depth"
# method_="conv"
# method_="conv_hyper_param_lr5_nwd"
# method_="late"
# method_="resnet"
# method_="resnet_fusion"
# method_="mlp"
# method_="method_2" # dont forget to change init in model llava_llama
# method_=dino
# method_=dino_late




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

    CKPT="VSR_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
    depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

elif [ "$dataset" = "VSR_BCE" ]; then
    train_file="VSR_train_TF.json"
    val_file="VSR_val_TF.json"

    CKPT="VSR_BCE_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
    depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

elif [ "$dataset" = "VSR_random" ]; then
    train_file="VSR_random_train_TF.json"
    val_file="VSR_random_validation_TF.json"

    CKPT="VSR_random_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
    depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

# elif [ "$dataset" = "VSR_class" ]; then
#     train_file="VSR_train_classification.json"
#     val_file="VSR_val_classification.json"

#     CKPT="VSR_class_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

#     depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
#     depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

elif [ "$dataset" = "VSR_random_class" ]; then
    train_file="VSR_random_train_classification.json"
    val_file="VSR_random_validation_classification.json"

    CKPT="VSR_random_class_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
    depth_val_file="/project/msc-thesis-project/all_vsr_depth/"


elif [ "$dataset" = "VSR_class2" ]; then
    train_file="VSR_train_classification2.json"
    val_file="VSR_val_classification2.json"

    CKPT="VSR_class2_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
    depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

elif [ "$dataset" = "instruct_37k_VSR_class2" ]; then
    train_file="instruct_37k_and_VSR_class2.json"
    val_file="VSR_val_classification2.json"

    CKPT="instruct_37k_VSR_class2_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
    depth_val_file="/project/msc-thesis-project/vall_vsr_depth/"

elif [ "$dataset" = "instruct_37k_VSR_class2_VSR_TF" ]; then
    train_file="instruct_37k_and_VSR_class2_VSR_TF.json"
    # train_file="llava_instruct_37k.json"
    val_file="VSR_val_TF.json"

    CKPT="instruct_37k_VSR_class2_VSR_TF_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/depth_instruct_150k_and_VSR_class2_VSR_TF/train/" # because i tried to download all but only got to 37k
    # depth_train_file="/project/msc-thesis-project/vsr_depth/train/"
    depth_val_file="/project/msc-thesis-project/vall_vsr_depth/"

# VSR* plus TF
elif [ "$dataset" = "VSR_combi" ]; then
    train_file="VSR_train_classification_combi_shuffled.json"
    val_file="VSR_val_classification_combi_shuffled.json" 

    CKPT="VSR_combi_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
    depth_val_file="/project/msc-thesis-project/all_vsr_depth/"


elif [ "$dataset" = "VSR_midas" ]; then
    train_file="VSR_train_TF_new.json"
    val_file="VSR_val_TF_new.json"

    CKPT="VSR_midas_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    # TODO: change gd to midas in all file names
    depth_train_file="/project/msc-thesis-project/vsr_depth_gd/train/"
    depth_val_file="/project/msc-thesis-project/vsr_depth_gd/val/"

############################################ What UP #######################################
elif [ "$dataset" = "Whatsup" ]; then
    train_file="whatsup_train_classification_controlled_images.json"
    val_file="whatsup_val_classification_controlled_images.json"

    CKPT="Whatsup_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"
    depth_val_file="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"

#################################### Spatialvlm ################################################
    
elif [ "$dataset" = "Spatialvlm" ]; then
    train_file="spatialvlm_train.json"
    val_file="spatialvlm_val.json"

    CKPT="Spatialvlm_epoch${epochs}-${method_}_${train_batchsize}_${eval_batchsize}_${mm_projector_lr}_${lr}-${model}"

    depth_train_file="/project/msc-thesis-project/spatialvlm/depth_images"
    depth_val_file="/project/msc-thesis-project/spatialvlm/depth_images"

####################################################################################

# train_file="whatsup_train_classification_COCO.json"
# fusion_method_="train.py"
# CKPT="WhatsupCOCO_epoch3-nodepth-llava-v1.5-7b"
# # CKPT="WhatsupCOCO_epoch3-nodepth-llava-v1.5-13b"

########################################## Different Model ###############################

# train_file="VSR_train_TF_copy.json"
# val_file="VSR_val_TF_copy.json"

# # train_file="whatsup_train_classification_controlled_images.json"
# # val_file="whatsup_val_classification_controlled_images.json"

# fusion_method_="train.py"
# model_base="Salesforce/instructblip-vicuna-7b"
# CKPT="VSR_TF_epoch3-nodepth-instructionblip_testing"

###########################################################################################
    # --deepspeed ./scripts/zero3.json \

fi

if [ "$method_" = "no_depth" ]; then
    fusion_method_="train.py"
elif [ "$method_" = "conv" ]; then
    fusion_method_="train_custom.py"
elif [ "$method_" = "mlp" ]; then
    fusion_method_="train_custom_mlp.py"
elif [ "$method_" = "method_2" ]; then
    fusion_method_="train_custom_method_2.py"
elif [ "$method_" = "late" ]; then
    fusion_method_="train_custom_late.py"
elif [ "$method_" = "resnet" ]; then
    fusion_method_="train_custom_resnet.py"
elif [ "$method_" = "resnet_fusion" ]; then
    fusion_method_="train_custom_resnet.py"
elif [ "$method_" = "de" ]; then
    fusion_method_="train_custom_de.py"
elif [ "$method_" = "dino" ]; then
    fusion_method_="train_custom_dino.py"
elif [ "$method_" = "dino_late" ]; then
    fusion_method_="train_custom_dino_late.py"
else
    echo "Invalid method_ specified."
    # fusion_method_="train_custom.py"
fi


if [[ "$dataset" == *VSR* ]]; then
    image_folder="/project/msc-thesis-project/all_vsr_images/"
else
    image_folder=""
fi

# -deepspeed ./scripts/zero3.json \

# --mm_use_im_patch_token False \
# report_to="tensorboard"

echo "Finetuning $model_base with $fusion_method_ on $train_file for $epochs epochs to become $CKPT"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# CUDA_VISIBLE_DEVICES=0 deepspeed /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method_ \
# python /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method_ \
# deepspeed /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method_ \
# CUDA_VISIBLE_DEVICES=0 python /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method_ \
CUDA_VISIBLE_DEVICES=$device python /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method_ \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr $mm_projector_lr \
    --model_name_or_path $model_base \
    --version v1 \
    --data_path /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${train_file} \
    --validation_data_path /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${val_file} \
    --depth_path_train "${depth_train_file}" \
    --depth_path_val "${depth_val_file}" \
    --image_folder "${image_folder}"\
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer $mm_vision_select_layer \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/checkpoint-${CKPT}-lora \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size $train_batchsize \
    --per_device_eval_batch_size $eval_batchsize \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --warmup_ratio $warmup_ratio \
    --lr_scheduler_type $lr_scheduler_type \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb\
    --run_name $CKPT
