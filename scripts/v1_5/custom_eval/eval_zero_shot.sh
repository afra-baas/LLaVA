#!/bin/bash
# question-file, dit is de input file in die jsonl format van LLaVA, of json
# answers-file , hier worden de model answers geoutput

device=1

gpu_list="${CUDA_VISIBLE_DEVICES:-$device}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

version="7b"

# dataset="Spatialvlm"
# dataset="VSR_combi"
# dataset="VSR"
# dataset="VSR_random"
# dataset="VSR_random_class"
# dataset="VSR_class"
dataset="VSR_class4"
# dataset="Whatsup"

method="zeroshot"
TF=False

if [ "$version" = "7b" ]; then
    model_base="liuhaotian/llava-v1.5-7b"
    model="llava-v1.5-7b"
elif [ "$version" = "13b" ]; then
    model_base="liuhaotian/llava-v1.5-13b"
    model="llava-v1.5-13b"
fi

################################## VSR ###################################################

if [ "$dataset" = "VSR" ]; then
    TF=True

    CKPT="VSR_TF_${method}-${model}"

    # depth_path="vsr" 
    depth_path="/project/msc-thesis-project/all_vsr_depth" 
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_TF"
    test_file="VSR_test_TF.json"

elif [ "$dataset" = "VSR_random" ]; then
    TF=True

    CKPT="VSR_random_TF_${method}-${model}"

    depth_path="/project/msc-thesis-project/all_vsr_depth" 
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_random_TF"
    test_file="VSR_random_test_TF.json"

elif [ "$dataset" = "VSR_class" ]; then
    CKPT="VSR_${method}-${model}"

    depth_path="vsr" 
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR"
    test_file="VSR_test_classification.json"


elif [ "$dataset" = "VSR_random_class" ]; then
    CKPT="VSR_random_${method}-${model}"

    depth_path="/project/msc-thesis-project/all_vsr_depth" 
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_random"
    test_file="VSR_random_test_classification.json"


elif [ "$dataset" = "VSR_class4" ]; then
    CKPT="VSR4_${method}-${model}"

    depth_path="vsr" 
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR4"
    test_file="VSR_test_classification4.json"

elif [ "$dataset" = "VSR_combi" ]; then
    CKPT="VSR_plus-${method}-${model}"

    depth_path="vsr" 
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_plus"
    test_file="VSR_test_classification_combi_shuffled.json"


################################## Whats Up ###################################################
elif [ "$dataset" = "Whatsup" ]; then
    CKPT="${method}-${model}"

    depth_path="/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images/"
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/whatsup"
    test_file="whatsup_test_classification_controlled_images.json"

#################################### Spatialvlm ######################################################

elif [ "$dataset" = "Spatialvlm" ]; then
    CKPT="Spatialvlm_${method}-${model}"

    depth_path="/project/msc-thesis-project/Spatialvlm/depth_images/"
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/Spatialvlm"
    test_file="spatialvlm_dataset.json"



######################################################################################################
# root='/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/whatsupCOCO'
# test_file="whatsup_val_classification_COCO.json"

####################################################################################################
fi

eval_file="llava.eval.model_vqa_science" 

echo "Evaluating $CKPT on $test_file with $eval_file with $TF"

# Create directory if it doesn't exist
mkdir -p "$(dirname "$root/$CKPT/")"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing IDX: $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $eval_file \
        --model-path $model_base \
        --question-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
        --image-folder '' \
        --depth-path $depth_path \
        --answers-file $root/$CKPT/${CHUNKS}_${IDX}_zeroshot.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$root/$CKPT/merge_zeroshot.jsonl

# Create directory if it doesn't exist
mkdir -p "$(dirname "$output_file")"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $root/$CKPT/${CHUNKS}_${IDX}_zeroshot.jsonl >> "$output_file"
done


# python scripts/convert_custom_for_submission.py \  convert_custom_for_submission_spatialvlm

# Evaluate
python scripts/convert_custom_for_submission.py \
    --annotation-file /project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/${test_file} \
    --result-file $output_file \
    --result-upload-file $root/${CKPT}/zeroshot_predictions.jsonl\
    --TF $TF

echo "Evaluating $CKPT on $test_file with $eval_file with $TF"
    