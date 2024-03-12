#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
CKPT="epoch3-nodepth-llava-v1.5-13b"
root='/project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/whatsupCOCO'

# question-file, dit is de input file in die jsonl format van LLaVA, of json
# answers-file , hier worden de model answers geoutput
# llava.eval.model_vqa_loader -> llava.eval.model_vqa_science
# --model-path liuhaotian/llava-v1.5-13b \

test_file="whatsup_val_classification_COCO.json"

# Create directory if it doesn't exist
mkdir -p "$(dirname "$root/$CKPT/")"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing IDX: $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-path ./checkpoints/checkpoint-WhatsupCOCO_${CKPT}-lora \
        --model-base liuhaotian/llava-v1.5-13b \
        --question-file /project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
        --image-folder '' \
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
    --annotation-file /project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
    --result-file $output_file \
    --result-upload-file $root/${CKPT}/predictions.jsonl

    