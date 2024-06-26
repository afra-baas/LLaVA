#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/data"

# CKPT="llava-v1.5-7b"
# --model-path liuhaotian/$CKPT \

# method='conv'
dataset='Whatsup'
base="llava-v1.5-7b"
device=1

# for method in 'dino' 'imagebind' 'imagebind_intermediate' ; do
# for method in 'imagebind' ; do
# for method in 'no_depth' 'conv' 'dino' 'imagebind' 'imagebind_intermediate' 'late' ; do
# for method in 'imagebind' 'imagebind_intermediate' 'late' ; do
for method in 'conv' 'dino' ; do
    # CKPT="${dataset}_epoch3-${method}-${base}_train_only_lin_proj"
    CKPT="${dataset}_epoch3-${method}-${base}"
        # --model-base liuhaotian/$base \
        # --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
    # CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \

    if [ "$method" = "no_depth" ] || [ -z "$method" ]; then
        eval_file=llava.eval.model_vqa_loader
    else
        eval_file=llava.eval.model_vqa_loader_$method
    fi

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=$device python -m $eval_file \
            --model-base liuhaotian/$base \
            --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
            --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/$SPLIT.jsonl \
            --image-folder /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/images/images \
            --depth-image-folder /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/depth_images \
            --answers-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    output_file=/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

    cd $GQADIR 
    python eval.py --tier testdev_balanced --output-dir /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/GQA_acc.jsonl

    echo /playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

done
