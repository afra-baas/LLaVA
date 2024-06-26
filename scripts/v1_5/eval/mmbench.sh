#!/bin/bash

SPLIT="mmbench_dev_20230712"
# CKPT='llava-v1.5-7b'
# --model-path liuhaotian/$CKPT \


# method='dino'
dataset='Whatsup'
base="llava-v1.5-7b"
    # --model-base liuhaotian/$base \
    # --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-$CKPT-lora \



for method in 'conv' 'dino' 'imagebind' 'imagebind_intermediate' 'late' ; do
# for method in 'imagebind' ; do

    # CKPT="${dataset}_epoch3-$method-${base}_train_only_lin_proj"
    CKPT="${dataset}_epoch3-$method-${base}"

    if [ "$method" = "no_depth" ] || [ -z "$method" ]; then
        eval_file=llava.eval.model_vqa_mmbench
    else
        eval_file=llava.eval.model_vqa_mmbench_$method
    fi

    # path="/project/msc-thesis-project/forked_repos/LLaVA/checkpoints_copy2/checkpoint-${CKPT}-lora"
    # if [ -d "$path" ]; then
    # echo "The path exists."
    # else
    # echo "The path does not exist."
    # fi

    python -m $eval_file \
        --model-base liuhaotian/$base \
        --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-$CKPT-lora \
        --depth-image-folder /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/depth_images/{}.jpg \
        --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/$CKPT/merge.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
        --result-dir /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/answers/$SPLIT \
        --upload-dir /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT \
        --experiment $CKPT

    python playground/data/eval/mmbench/eval.py \
        --upload-dir /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT/$CKPT.xlsx \
        --output-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/$CKPT/mmb_acc.jsonl \

    echo /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT/$CKPT.xlsx

done