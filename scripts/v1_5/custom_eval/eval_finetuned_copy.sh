#!/bin/bash
# question-file, dit is de input file in die jsonl format van LLaVA, of json
# answers-file , hier worden de model answers geoutput


device=1

gpu_list="${CUDA_VISIBLE_DEVICES:-$device}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


epochs=10
version="7b"

dataset="VSR"
# dataset="VSR_class2"
# dataset="VSR_class"
# dataset="Whatsup"
# dataset="VSR_midas"

# method="no_depth"
# method="conv_hyper_param_patch_start_lr3"
# method="conv_hyper_param_lr5"
# method="late"
# method="resnet"
# method="conv"
# method="mlp"
# method="method2" # dont forget to change init in model llava_llama
method="dino"

TF=False


if [ "$version" = "7b" ]; then
    model_base="liuhaotian/llava-v1.5-7b"
    model="llava-v1.5-7b"
elif [ "$version" = "13b" ]; then
    model_base="liuhaotian/llava-v1.5-13b"
    model="llava-v1.5-13b"
fi


CKPT="VSR_epoch${epochs}-${method}-${model}"
depth_path="vsr" 
root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_FB"
test_file="clevr_front_behind_test.json"



if [ "$method" = "no_depth" ]; then
    eval_file="llava.eval.model_vqa_science" 
elif [ "$method" = "conv" ]; then
    eval_file="llava.eval.model_vqa_science_depth" 
elif [ "$method" = "mlp" ]; then
    eval_file="llava.eval.model_vqa_science_depth_mlp" 
elif [ "$method" = "method2" ]; then
    eval_file="llava.eval.model_vqa_science_method2" 
elif [ "$method" = "late" ]; then
    eval_file="llava.eval.model_vqa_science_late"
elif [ "$method" = "resnet" ]; then
    eval_file="llava.eval.model_vqa_science_resnet"
elif [ "$method" = "dino" ]; then
    eval_file="llava.eval.model_vqa_science_dino"
else
    echo "Invalid method specified."
    # eval_file="llava.eval.model_vqa_science_depth"
fi


echo "Evaluating $CKPT on $test_file with $eval_file"
# Create directory if it doesn't exist
mkdir -p "$(dirname "$root/$CKPT/")"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing IDX: $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $eval_file \
        --model-path ./checkpoints/checkpoint-${CKPT}-lora \
        --model-base $model_base \
        --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
        --image-folder '' \
        --depth-path $depth_path \
        --answers-file $root/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

output_file=$root/$CKPT/merge.jsonl

# Create directory if it doesn't exist
mkdir -p "$(dirname "$output_file")"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $root/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_custom_for_submission.py \
    --annotation-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
    --result-file $output_file \
    --result-upload-file $root/${CKPT}/predictions.jsonl\
    --TF $TF

echo "Evaluating $CKPT on $test_file with $eval_file"


    