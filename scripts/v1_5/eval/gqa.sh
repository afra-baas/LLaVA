#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-13b"
SPLIT="llava_gqa_testdev_balanced"
# GQADIR="./playground/data/eval/gqa/data"
# --model-path liuhaotian/$CKPT \

method='no_depth'
dataset='VSR'

base="llava-v1.5-7b"
CKPT="${dataset}_epoch3-${method}-${base}"
GQADIR="./playground/data/eval/gqa/data/$CKPT"

        # --model-base liuhaotian/$base \
        # --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
# CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \

device=1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$device python -m llava.eval.model_vqa_loader \
        --model-base liuhaotian/$base \
        --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR 
python eval.py --tier testdev_balanced --output-dir GQA_acc_${dataset}_${method}_${base}.jsonl

echo /playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl
