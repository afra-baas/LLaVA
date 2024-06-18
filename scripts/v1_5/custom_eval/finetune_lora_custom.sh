#!/bin/bash
# llava/train/train_mem.py \
# only one of VSR or Whatsup on at the time

export WANDB_API_KEY='6349ed9b917e6fee85da5729128e382f85ee0f53'
export WANDB_PROJECT="Thesis_runs"

# WANDB_MODE=offline
epochs=3
version="7b"
device=0
train_batchsize=16
eval_batchsize=16
weight_decay=0.
lr=2e-4
warmup_ratio=0.03
lr_scheduler_type="cosine"
mm_vision_select_layer=-2
mm_projector_lr=2e-5
# mm_projector_lr=2e-2

# dataset="VSR"
# dataset="VSR_f"
# dataset="VSR_BCE"
# dataset="VSR_class4"
# dataset="VSR_combi"
# dataset="VSR_class"
# dataset="Whatsup"
# dataset="VSR_midas"
# dataset="instruct_37k_VSR_class2_VSR_TF"
# dataset="instruct_37k_VSR_class2"
# dataset="Spatialvlm"
# dataset="VSR_random"
# dataset="VSR_random_BCE"
# dataset="VSR_random_class"


# method="no_depth"
# method="conv"
# method="late"
# method="resnet"
# method="resnet_fusion"
# method="mlp"
# method="method2" # dont forget to change init in model llava_llama, i now import in train_custom
# method=dino
# method=dino_late
# method="imagebind"
method="imagebind_intermediate"


# for method in 'conv' 'imagebind' ; do
# for method in 'imagebind_intermediate' ; do
for dataset in  "VSR_class4" "VSR_f" "VSR_random" ; do
# for dataset in "VSR_random" ; do
# for dataset in "VSR" "VSR_f" "Whatsup" ; do
# for dataset in "VSR_f"  ; do
# for dataset in "VSR_class4" ; do
# for dataset in "Whatsup" ; do

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

        CKPT="VSR_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_epoch${epochs}-${method}-${model}_freeze_dino"
        # CKPT="VSR_epoch${epochs}-${method}-${model}_only_lin_proj"
        # CKPT="VSR_plus_BCE_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_epoch${epochs}-${method}-${model}_train_only_lin_proj"

        depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
        depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

    elif [ "$dataset" = "VSR_f" ]; then
        train_file="VSR_train_TF_f.json"
        val_file="VSR_val_TF_f.json"

        # CKPT="VSR_f_epoch${epochs}-${method}-${model}_train_only_lin_proj"
        CKPT="VSR_f_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_f_epoch${epochs}-${method}-${model}_freeze_dino"
        # CKPT="VSR_f_epoch${epochs}-${method}-${model}_only_lin_proj"


        depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
        depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

    elif [ "$dataset" = "VSR_BCE" ]; then
        train_file="VSR_train_TF.json"
        val_file="VSR_val_TF.json"

        CKPT="VSR_BCE_TF_epoch${epochs}-${method}-${model}"

        depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
        depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

    elif [ "$dataset" = "VSR_random" ]; then
        train_file="VSR_random_train_TF.json"
        val_file="VSR_random_validation_TF.json"

        # CKPT="VSR_random_epoch${epochs}-${method}-${model}_train_only_lin_proj"
        CKPT="VSR_random_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_epoch${epochs}-${method}-${model}_freeze_dino"

        depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
        depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

    elif [ "$dataset" = "VSR_class" ]; then
        train_file="VSR_train_classification.json"
        val_file="VSR_val_classification.json"

        CKPT="VSR_class_epoch${epochs}-${method}-${model}"

        depth_train_file="/project/msc-thesis-project/vsr_depth/train/"
        depth_val_file="/project/msc-thesis-project/vsr_depth/val/"

    elif [ "$dataset" = "VSR_random_class4" ]; then
        train_file="VSR_random_train_classification4.json"
        val_file="VSR_random_validation_classification4.json"

        # CKPT="VSR_random_class4_epoch${epochs}-${method}-${model}_train_only_lin_proj"
        CKPT="VSR_random_class4_epoch${epochs}-${method}-${model}"

        depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
        depth_val_file="/project/msc-thesis-project/all_vsr_depth/"


    elif [ "$dataset" = "VSR_class4" ]; then
        train_file="VSR_train_classification4.json"
        val_file="VSR_val_classification4.json"

        # CKPT="VSR_class4_epoch${epochs}-${method}-${model}_unfrozen_all"
        # CKPT="VSR4_epoch${epochs}-${method}-${model}_frozen_llm"
        # CKPT="VSR4_epoch${epochs}-${method}-${model}_train_only_conv"
        # CKPT="VSR4_epoch${epochs}-${method}-${model}_only_proj2"
        # CKPT="VSR_class4_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_class4_epoch${epochs}-${method}-${model}_no_finetuing"
        # CKPT="VSR_class4_epoch${epochs}-${method}-${model}_train_only_lin_proj"
        CKPT="VSR_class4_epoch${epochs}-${method}-${model}"

        depth_train_file="/project/msc-thesis-project/all_vsr_depth/"
        depth_val_file="/project/msc-thesis-project/all_vsr_depth/"

    elif [ "$dataset" = "instruct_37k_VSR_class2" ]; then
        train_file="instruct_37k_and_VSR_class2.json"
        val_file="VSR_val_classification2.json"

        CKPT="instruct_37k_VSR_class2_epoch${epochs}-${method}-${model}"

        depth_train_file="/project/msc-thesis-project/vsr_depth/train/"
        depth_val_file="/project/msc-thesis-project/vsr_depth/val/"

    elif [ "$dataset" = "instruct_37k_VSR_class2_VSR_TF" ]; then
        train_file="instruct_37k_and_VSR_class2_VSR_TF.json"
        # train_file="llava_instruct_37k.json"
        val_file="VSR_val_TF.json"

        CKPT="instruct_37k_VSR_class2_VSR_TF_epoch${epochs}-${method}-${model}"

        depth_train_file="/project/msc-thesis-project/depth_instruct_150k_and_VSR_class2_VSR_TF/train/" # because i tried to download all but only got to 37k
        # depth_train_file="/project/msc-thesis-project/vsr_depth/train/"
        depth_val_file="/project/msc-thesis-project/vsr_depth/val/"

    # VSR* plus TF
    elif [ "$dataset" = "VSR_combi" ]; then
        train_file="VSR_train_classification_combi_shuffled.json"
        val_file="VSR_val_classification_combi_shuffled.json" 

        CKPT="VSR_combi_epoch${epochs}-${method}-${model}"

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

        # CKPT="Whatsup_epoch${epochs}-${method}-${model}_train_only_lin_proj"
        CKPT="Whatsup_epoch${epochs}-${method}-${model}"
        # CKPT="Whatsup_epoch${epochs}-${method}-${model}_freeze_dino"
        # CKPT="Whatsup_epoch${epochs}-${method}-${model}_only_lin_proj"


        depth_train_file="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"
        depth_val_file="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"

    #################################### Spatialvlm ################################################
        
    elif [ "$dataset" = "Spatialvlm" ]; then
        train_file="spatialvlm_train.json"
        val_file="spatialvlm_val.json"

        CKPT="Spatialvlm_epoch${epochs}-${method}-${model}"

        depth_train_file="/project/msc-thesis-project/spatialvlm/depth_images"
        depth_val_file="/project/msc-thesis-project/spatialvlm/depth_images"

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

    fi

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
    elif [ "$method" = "dino_late" ]; then
        fusion_method="train_custom_dino_late.py"
    elif [ "$method" = "fmask" ]; then
        fusion_method="train_custom_fmask.py"
    elif [ "$method" = "imagebind" ]; then
        fusion_method="train_custom_imagebind.py"
    elif [ "$method" = "imagebind_intermediate" ]; then
        fusion_method="train_custom_imagebind_intermediate.py"
    else
        echo "Invalid method specified."
        # fusion_method="train_custom.py"
    fi 


    if [[ "$dataset" == *VSR* ]]; then
        image_folder="/project/msc-thesis-project/all_vsr_images/"
    else
        image_folder=""
    fi

    # -deepspeed ./scripts/zero3.json \

    # --mm_use_im_patch_token False \
    # --evaluation_strategy "epoch" \

    #     --vision_tower_lr 2e-5 \
    #     --vit_lora_enable \
    #     --lora_alpha_vit 128 \
    #     --lora_r_vit 64 \
    # --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    # --pretrain_mm_mlp_adapter liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    
        # --pretrain_mm_mlp_adapter $model_base/mm_projector.bin \

    echo "Finetuning $model_base with $fusion_method on $train_file for $epochs epochs to become $CKPT"

    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    # CUDA_VISIBLE_DEVICES=0 deepspeed /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method \
    # python /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method \
    # deepspeed /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method \
    # CUDA_VISIBLE_DEVICES=0 python /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method \
    CUDA_VISIBLE_DEVICES=$device python /project/msc-thesis-project/forked_repos/LLaVA/llava/train/$fusion_method \
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

done
