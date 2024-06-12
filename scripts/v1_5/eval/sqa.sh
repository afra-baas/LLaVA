#!/bin/bash

base_model=llava-v1.5-7b

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/${base_model} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${base_model}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${base_model}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${base_model}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${base_model}_result.json
