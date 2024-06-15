#!/bin/bash
# question-file, dit is de input file in die jsonl format van LLaVA, of json
# answers-file , hier worden de model answers geoutput


device=1

gpu_list="${CUDA_VISIBLE_DEVICES:-$device}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


epochs=3
version="7b"

# dataset="VSR"
# dataset="VSR_class4"
# dataset="VSR_class"
# dataset="Whatsup"
# dataset="VSR_midas"
# dataset="clevr_front_behind"
# dataset="clevr"
# dataset="VSR_random"
# dataset="VSR_random_class"
# dataset="Spatialvlm"

# method="no_depth"
# method="conv_hyper_param_patch_start_lr3"
# method="conv_hyper_param_lr5"
# method="late"
# method="resnet"
# method="conv"
# method="mlp"
# method="method2" # dont forget to change init in model llava_llama
# method="dino"
# method="dino_late"
method="imagebind"

# "no_depth" "conv" "dino" "late"

# for method in "imagebind" ; do
# for dataset in "VSR" "VSR_f" "Whatsup" ; do
# for dataset in  "VSR_random" ; do
for dataset in  "VSR" ; do

    TF="False"

    if [ "$version" = "7b" ]; then
        model_base="liuhaotian/llava-v1.5-7b"
        model="llava-v1.5-7b"
    elif [ "$version" = "13b" ]; then
        model_base="liuhaotian/llava-v1.5-13b"
        model="llava-v1.5-13b"
    fi
    ################################## VSR ###################################################
    if [ "$dataset" = "VSR" ]; then
        TF="True"
        # CKPT="VSR_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_class_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_BCE_TF_epoch${epochs}-${method}-${model}"
        # CKPT="Spatialvlm_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_plus_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_TF_plus_BCE_epoch${epochs}-${method}-${model}"
        CKPT="VSR_epoch${epochs}-${method}-${model}_train_only_lin_proj"



        # CKPT="VSR_class4_epoch${epochs}-${method}-${model}"
        # CKPT="VSR2_epoch${epochs}-${method}-${model}"
        # CKPT="epoch${epochs}-${method}-${model}"

        # depth_path="vsr"
        depth_path="/project/msc-thesis-project/all_vsr_depth" 
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR"
        test_file="VSR_test_TF.json"
        # test_file="VSR_test_TF_for_whatsup.json"

    elif [ "$dataset" = "VSR_f" ]; then
        TF="True"

        CKPT="VSR_f_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_f_epoch${epochs}-${method}-${model}_train_only_lin_proj"

        depth_path="/project/msc-thesis-project/all_vsr_depth" 
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_f"
        test_file="VSR_test_TF_f.json"

    elif [ "$dataset" = "VSR_random" ]; then
        TF="True"

        CKPT="VSR_random_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_TF_epoch${epochs}-${method}-${model}"
        # CKPT="VSR2_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_class4_epoch${epochs}-${method}-${model}"
        # CKPT="epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_epoch${epochs}-${method}-${model}_train_only_lin_proj"

        depth_path="/project/msc-thesis-project/all_vsr_depth" 
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_random"
        test_file="VSR_random_test_TF.json"
        # test_file="VSR_random_test_TF_for_whatsup.json"


    elif [ "$dataset" = "VSR_class" ]; then
        # CKPT="VSR_epoch${epochs}-${method}-${model}"
        CKPT="VSR_class_epoch${epochs}-${method}-${model}"

        # depth_path="vsr" 
        depth_path="/project/msc-thesis-project/all_vsr_depth" 
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_class"
        test_file="VSR_test_classification.json"
        # test_file="COCO_classification_front_behind_data_with_obj_depths.json"

    elif [ "$dataset" = "VSR_random_class" ]; then
        CKPT="VSR_random_class_epoch${epochs}-${method}-${model}"
        # CKPT="VSR2_epoch${epochs}-${method}-${model}"
        # CKPT="epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_TF_epoch${epochs}-${method}-${model}"


        depth_path="/project/msc-thesis-project/all_vsr_depth/" 
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_random_class"
        test_file="VSR_random_test_classification.json"


    elif [ "$dataset" = "VSR_class4" ]; then
        CKPT="VSR_class4_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_class4_epoch${epochs}-${method}-${model}_train_only_imagebind"
        # CKPT="VSR_class4_epoch${epochs}-${method}-${model}_train_only_lin_proj"
        # CKPT="VSR4_epoch${epochs}-${method}-${model}_train_only_conv"
        # CKPT="VSR4_epoch${epochs}-${method}-${model}_only_proj2"

        # CKPT="VSR4_epoch${epochs}-${method}-${model}_unfrozen_all"

        # CKPT="VSR_random_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_TF_epoch${epochs}-${method}-${model}"
        # CKPT="epoch${epochs}-${method}-${model}"
        # CKPT="VSR_plus_epoch${epochs}-${method}-${model}"

        # depth_path="vsr" 
        depth_path="/project/msc-thesis-project/all_vsr_depth" 
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_class4"
        test_file="VSR_test_classification4.json"

    elif [ "$dataset" = "VSR_midas" ]; then
        CKPT="VSR_midas_epoch${epochs}-${method}-${model}"

        # depth_path="vsr"
        depth_path="/project/msc-thesis-project/all_vsr_depth" 
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_midas"
        test_file="VSR_test_TF_new.json"


    ################################## Whats Up ###################################################
    elif [ "$dataset" = "Whatsup" ]; then
        CKPT="Whatsup_epoch${epochs}-${method}-${model}"
        # CKPT="Spatialvlm_epoch${epochs}-${method}-${model}"
        # CKPT="VSR2_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_random_TF_epoch${epochs}-${method}-${model}"
        # CKPT="VSR_TF_epoch${epochs}-${method}-${model}"
        # CKPT="Whatsup_epoch${epochs}-${method}-${model}_train_only_lin_proj"


        depth_path="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/Whatsup"
        test_file="whatsup_test_classification_controlled_images.json"

    ######################################################################################################
    elif [ "$dataset" = "clevr" ]; then
        CKPT="Whatsup_epoch${epochs}-${method}-${model}"
        # CKPT="VSR2_epoch${epochs}-${method}-${model}"

        depth_path="/project/clevr_depth/test"
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/clevr"
        test_file="clevr_test.json"

    elif [ "$dataset" = "clevr_front_behind" ]; then
        # CKPT="Whatsup_epoch${epochs}-${method}-${model}"
        # CKPT="VSR2_epoch${epochs}-${method}-${model}"


        depth_path="/project/clevr_depth/test"
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/clevr_bf"
        test_file="clevr_front_behind_test.json"


    #################################### Spatialvlm ######################################################

    elif [ "$dataset" = "Spatialvlm" ]; then
        CKPT="Spatialvlm_${method}-${model}"

        depth_path="/project/msc-thesis-project/Spatialvlm/depth_images/"
        root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/Spatialvlm"
        test_file="spatialvlm_dataset.json"


    ######################################################################################################

    # CKPT="WhatsupCOCO_epoch3-nodepth-llava-v1.5-7b"
    # CKPT="WhatsupCOCO_epoch3-nodepth-llava-v1.5-13b"
    # root='/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/whatsupCOCO'
    # test_file="whatsup_val_classification_COCO.json"

    ####################################################################################################
    fi

    if [ "$method" = "no_depth" ]; then
        eval_file="llava.eval.model_vqa_science" 
    elif [ "$method" = "conv" ]; then
        eval_file="llava.eval.model_vqa_science_depth" 
    elif [ "$method" = "mlp" ]; then
        eval_file="llava.eval.model_vqa_science_depth_mlp" 
    elif [ "$method" = "method2" ]; then
        eval_file="llava.eval.model_vqa_science_method2" 
    elif [ "$method" = "late" ]; then
        eval_file="llava.eval.model_vqa_science_late"
    elif [ "$method" = "resnet" ]; then
        eval_file="llava.eval.model_vqa_science_resnet"
    elif [ "$method" = "dino" ]; then
        eval_file="llava.eval.model_vqa_science_dino"
    elif [ "$method" = "dino_late" ]; then
        eval_file="llava.eval.model_vqa_science_dino_late"
    elif [ "$method" = "fmask" ]; then
        eval_file="llava.eval.model_vqa_science_fmask"
    elif [ "$method" = "imagebind" ]; then
        eval_file="llava.eval.model_vqa_science_imagebind"
    else
        echo "Invalid method specified."
        # eval_file="llava.eval.model_vqa_science_depth"
    fi


    echo "Evaluating $CKPT on $test_file with $eval_file"
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$root/$CKPT/")"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        echo "Processing IDX: $IDX"
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $eval_file \
            --model-path ./checkpoints/checkpoint-${CKPT}-lora \
            --model-base $model_base \
            --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
            --image-folder '' \
            --depth-path $depth_path \
            --answers-file $root/$CKPT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait

    echo "Done with answer generation"

    output_file=$root/$CKPT/merge.jsonl

    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "$root/$CKPT/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
    done

    # Evaluate
    python scripts/convert_custom_for_submission.py \
        --annotation-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
        --result-file $output_file \
        --result-upload-file $root/${CKPT}/predictions.jsonl\
        --TF $TF

    echo "Evaluating $CKPT on $test_file with $eval_file"

done

    