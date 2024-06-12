#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"
# --model-path liuhaotian/$CKPT \

method='conv'

base="llava-v1.5-7b"
CKPT="VSR_TF_epoch10-$method-${base}"
# CKPT="epoch3-$method-${base}"
eval_file=llava.eval.model_vqa_loader_$method

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $eval_file \
        --model-base liuhaotian/$base \
        --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images/images \
        --depth-image-folder ./playground/data/eval/gqa/depth_images \
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
python eval.py --tier testdev_balanced 

echo /playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl
