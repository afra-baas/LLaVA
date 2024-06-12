#!/bin/bash

version=$1
dataset=$2
device=$3

gpu_list="${CUDA_VISIBLE_DEVICES:-$device}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

method="zeroshot"

if [ "$version" = "7b" ]; then
    model_base="liuhaotian/llava-v1.5-7b"
    model="llava-v1.5-7b"
elif [ "$version" = "13b" ]; then
    model_base="liuhaotian/llava-v1.5-13b"
    model="llava-v1.5-13b"
fi

################################## VSR ###################################################

if [ "$dataset" = "VSR" ]; then
    CKPT="VSR_TF_${method}-${model}"

    depth_path="vsr" # not used in model_vqa_science
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_TF"
    test_file="VSR_test_TF.json"

################################## Whats Up ###################################################
elif [ "$dataset" = "Whatsup" ]; then
    CKPT="${method}-${model}"

    depth_path="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/whatsup"
    test_file="whatsup_test_classification_controlled_images.json"

######################################################################################################
# root='/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/whatsupCOCO'
# test_file="whatsup_val_classification_COCO.json"

####################################################################################################
fi

eval_file="llava.eval.model_vqa_science" 


echo "Evaluating $CKPT on $test_file"
# Create directory if it doesn't exist
mkdir -p "$(dirname "$root/$CKPT/")"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing IDX: $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $eval_file \
        --model-path $model_base \
        --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
        --image-folder '' \
        --depth-path $depth_path \
        --answers-file $root/$CKPT/${CHUNKS}_${IDX}_zeroshot.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$root/$CKPT/merge_zeroshot.jsonl

# Create directory if it doesn't exist
mkdir -p "$(dirname "$output_file")"


# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $root/$CKPT/${CHUNKS}_${IDX}_zeroshot.jsonl >> "$output_file"
done


# Evaluate
python scripts/convert_custom_for_submission.py \
    --annotation-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
    --result-file $output_file \
    --result-upload-file $root/${CKPT}/zeroshot_predictions.jsonl

echo "Evaluating $CKPT on $test_file with $eval_file"
    