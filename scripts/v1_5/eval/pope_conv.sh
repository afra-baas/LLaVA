#!/bin/bash

method='conv'

source='liuhaotian/'
base="llava-v1.5-7b"
CKPT="VSR_TF_epoch10-$method-${base}"
# CKPT="epoch3-$method-${base}"
eval_file=llava.eval.model_vqa_loader_$method
# --model-path ./checkpoints/checkpoint-${CKPT}-lora \


python -m $eval_file \
    --model-base $source$base \
    --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --depth-image-folder /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/pope/val2014_depth_images/ \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

echo ./playground/data/eval/pope/answers/$CKPT.jsonl
CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl

