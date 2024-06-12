#!/bin/bash
# CKPT="llava-v1.5-13b"
source='liuhaotian/'
# --model-path $source$CKPT \

model="llava-v1.5-7b"
CKPT="VSR_TF_epoch3-no_depth-${model}"
# CKPT="epoch3-no_depth-${model}"

    # --model-base $source$model \
    # --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \

python -m llava.eval.model_vqa_loader \
    --model-base $source$model \
    --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

echo ./playground/data/eval/pope/answers/$CKPT.jsonl
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl



