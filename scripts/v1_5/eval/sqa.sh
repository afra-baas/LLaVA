#!/bin/bash
# CKPT="llava-v1.5-7b"
# --model-path liuhaotian/$CKPT \

# method='dino'
dataset='Whatsup'
# dataset='VSR'
base="llava-v1.5-7b"

# for method in 'imagebind' 'late' 'imagebind_intermediate' ; do
# for method in 'no_depth' ; do
# for method in 'no_depth' 'conv' ; do
for method in 'conv' 'dino' 'imagebind' 'imagebind_intermediate' 'late' ; do
    # CKPT="${dataset}_epoch3-${method}-${base}_train_only_lin_proj"
    CKPT="${dataset}_epoch3-${method}-${base}"
    # --model-base liuhaotian/$base \
    # --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
    
    # path="/project/msc-thesis-project/forked_repos/LLaVA/checkpoints_copy2/checkpoint-${CKPT}-lora"
    # if [ -d "$path" ]; then
    # echo "The path exists."
    # else
    # echo "The path does not exist."
    # fi


    if [ "$method" = "no_depth" ] || [ -z "$method" ]; then
        eval_file=llava.eval.model_vqa_science
    else
        eval_file=llava.eval.model_vqa_science_$method
    fi

    echo $eval_file
    python -m $eval_file \
        --model-base liuhaotian/$base \
        --model-path /project/msc-thesis-project/forked_repos/LLaVA/checkpoints/checkpoint-${CKPT}-lora \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --depth-path ./playground/data/eval/scienceqa/depth_images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}/merge.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    python llava/eval/eval_science_qa.py \
        --base-dir ./playground/data/eval/scienceqa \
        --result-file ./playground/data/eval/scienceqa/answers/${CKPT}/merge.jsonl \
        --output-file ./playground/data/eval/scienceqa/answers/${CKPT}/output.jsonl \
        --output-result ./playground/data/eval/scienceqa/answers/${CKPT}/result.json 

done
