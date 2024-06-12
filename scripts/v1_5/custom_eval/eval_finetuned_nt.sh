#!/bin/bash

epochs=$1
version=$2
dataset=$3
method=$4
device=$5

gpu_list="${CUDA_VISIBLE_DEVICES:-$device}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


# dataset="VSR"
# dataset="Whatsup"
# dataset="VSR_midas"

# method="no_depth"
# method="conv_hyper_param_patch_start_lr3"
# method="conv_hyper_param_lr5"
# method="late"
# method="conv"
# method="mlp"
# method="method2" # dont forget to change init in model llava_llama

if [ "$version" = "7b" ]; then
    model_base="liuhaotian/llava-v1.5-7b"
    model="llava-v1.5-7b"
elif [ "$version" = "13b" ]; then
    model_base="liuhaotian/llava-v1.5-13b"
    model="llava-v1.5-13b"
fi

################################## VSR ###################################################
if [ "$dataset" = "VSR" ]; then
    CKPT="VSR_TF_epoch${epochs}-${method}-${model}"

    depth_path="vsr"
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_TF"
    test_file="VSR_test_TF.json"

elif [ "$dataset" = "VSR_class" ]; then
    CKPT="VSR_epoch${epochs}-${method}-${model}"

    depth_path="vsr" 
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR"
    test_file="VSR_test_classification.json"

elif [ "$dataset" = "VSR_midas" ]; then
    CKPT="VSR_TF_midas_epoch${epochs}-${method}-${model}"

    depth_path="vsr"
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_TF"
    test_file="VSR_test_TF_new.json"


################################## Whats Up ###################################################
elif [ "$dataset" = "Whatsup" ]; then
    CKPT="epoch${epochs}-${method}-${model}"

    depth_path="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/whatsup"
    test_file="whatsup_test_classification_controlled_images.json"

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
elif [ "$method" = "de" ]; then
    eval_file="llava.eval.model_vqa_science_de"
elif [ "$method" = "dino" ]; then
    eval_file="llava.eval.model_vqa_science_dino"
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

output_file=$root/$CKPT/merge.jsonl

# Create directory if it doesn't exist
mkdir -p "$(dirname "$output_file")"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $root/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_custom_for_submission.py \
    --annotation-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
    --result-file $output_file \
    --result-upload-file $root/${CKPT}/predictions.jsonl

echo "Evaluating $CKPT on $test_file with $eval_file"


    