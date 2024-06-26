#!/bin/bash
# CKPT="llava-v1.5-7b"
# --model-path liuhaotian/$CKPT \

# method='dino'
dataset='Whatsup'
base="llava-v1.5-7b"

# for method in 'conv' 'imagebind' 'late' 'imagebind_intermediate' ; do
# for method in 'imagebind' ; do
for method in 'dino' ; do
# for method in 'no_depth' 'conv' 'dino' 'imagebind' 'imagebind_intermediate' 'late' ; do
    # CKPT="${dataset}_epoch3-${method}-${base}_train_only_lin_proj"
    CKPT="${dataset}_epoch3-${method}-${base}"
        # --model-base liuhaotian/$base \
        # --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \

    if [ "$method" = "no_depth" ] || [ -z "$method" ]; then
        eval_file=llava.eval.model_vqa_loader
    else
        eval_file=llava.eval.model_vqa_loader_$method
    fi

    echo $eval_file
    python -m $eval_file \
        --model-base liuhaotian/$base \
        --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
        --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/val2014 \
        --depth-image-folder /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/val2014_depth_images/ \
        --answers-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/answers/$CKPT/merge.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    echo /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/answers/$CKPT/merge.jsonl
    python llava/eval/eval_pope.py \
        --annotation-dir /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/coco \
        --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
        --result-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/answers/$CKPT/merge.jsonl \
        --output-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/answers/$CKPT/pope_acc.jsonl 

done

