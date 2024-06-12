#!/bin/bash

SPLIT="mmbench_dev_20230712"
# CKPT='llava-v1.5-7b'
# --model-path liuhaotian/$CKPT \

method='mlp'

source='liuhaotian/'
base="llava-v1.5-7b"
CKPT="VSR_TF_epoch10-$method-${base}"
# CKPT="epoch3-$method-${base}"
eval_file=llava.eval.model_vqa_mmbench_$method

# CKPT="checkpoint-VSR_TF_epoch3-conv-${model}-lora"

python -m $eval_file \
    --model-base $source$base \
    --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
    --depth-image-folder /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/depth_images/{}.jpg \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT

python playground/data/eval/mmbench/eval.py \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT/$CKPT.xlsx \

echo ./playground/data/eval/mmbench/answers_upload/$SPLIT/$CKPT.xlsx