#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="epoch10-llava-v1.5-13b-coco_depth_front_behind"

# question-file, dit is de input file in die jsonl format van LLaVA, of json
# answers-file , hier worden de model answers geoutput
# llava.eval.model_vqa_loader -> llava.eval.model_vqa_science
# --model-path liuhaotian/$CKPT \

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-path ./checkpoints/checkpoint-epoch10-llava-v1.5-13b-lora_depth_front_behind \
        --model-base liuhaotian/llava-v1.5-13b \
        --question-file ./playground/data/eval/custom/COCO_test_classification_front_behind.json \
        --image-folder '' \
        --answers-file ./playground/data/eval/custom/answers_custom/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/custom/answers_custom/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/custom/answers_custom/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


# Evaluate
python scripts/convert_custom_for_submission.py \
    --annotation-file ./playground/data/eval/custom/COCO_test_classification_front_behind.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/custom/answers_upload/$CKPT.jsonl

    