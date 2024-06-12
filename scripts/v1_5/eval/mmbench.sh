#!/bin/bash

SPLIT="mmbench_dev_20230712"
# CKPT='llava-v1.5-13b'
# --model-path liuhaotian/$CKPT \

model="llava-v1.5-13b"
CKPT="VSR_TF_epoch3-no_depth-${model}"
# CKPT="epoch3-no_depth-${model}"

python -m llava.eval.model_vqa_mmbench \
    --model-base liuhaotian/$model \
    --model-path "/project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-$CKPT-lora" \
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
