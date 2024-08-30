#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
source ~/.bashrc
conda activate magmax

superglue_tasks=("wic" "cb" "rte" "copa" "boolq" "wsc" "multirc")

model_path="google-t5/t5-small"

# Loop through the shuffled list of tasks
for i in "${!superglue_tasks[@]}"; do
    task=${superglue_tasks[$i]}
    output_dir="exp/seq-fine/t5-small-$(IFS=-; echo "${superglue_tasks[*]:0:$((i+1))}")"

    python finetune.py --model_name_or_path "$model_path" --do_train --do_predict \
        --output_dir "$output_dir" --per_device_train_batch_size=32 --per_device_eval_batch_size=32 \
        --learning_rate=1e-4 --save_strategy "epoch" --overwrite_output_dir --lang "$task" \
        --evaluation_strategy "epoch" --num_train_epochs=10 --predict_with_generate --logging_first_step \
        --adafactor --load_best_model_at_end --metric_for_best_model "exact_match" --greater_is_better True
    
    # Remove checkpoints
    find "$output_dir" -type d -name 'checkpoint*' -exec rm -rf {} +
    
    # Update the model path for the next iteration
    model_path="$output_dir"
done

# Calc fisher information matrix
for i in "${!superglue_tasks[@]}"; do
    task=${superglue_tasks[$i]}
    model_path="exp/seq-fine/t5-small-$(IFS=-; echo "${superglue_tasks[*]:0:$((i+1))}")"
    output_dir="${model_path}-fim"
    echo $model_path
    echo $output_dir
    python fim.py --model_name_or_path "$model_path" --lang "$task" --output_dir "$output_dir"
done

#perform magmax and evaluate the model

# Loop from 0.1 to 1 in increments of 0.1
for i in $(seq 0.1 0.1 1.0); do
    # Format the number to one decimal place
    formatted_number=$(printf "%.1f" $i)
    
    folder_name="exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-${formatted_number}"
    mkdir -p "$folder_name"    
    python magmax.py --model1 exp/seq-fine/t5-small-wic --model2 exp/seq-fine/t5-small-wic-cb --model3 exp/seq-fine/t5-small-wic-cb-rte --model4 exp/seq-fine/t5-small-wic-cb-rte-copa --model5 exp/seq-fine/t5-small-wic-cb-rte-copa-boolq --model6 exp/seq-fine/t5-small-wic-cb-rte-copa-boolq-wsc --model7 exp/seq-fine/t5-small-wic-cb-rte-copa-boolq-wsc-multirc  --lamda $formatted_number --save_dir $folder_name

    python finetune.py --model_name_or_path $folder_name --do_predict --output_dir $folder_name  --per_device_eval_batch_size=32 --overwrite_output_dir --predict_with_generate

    rm "${folder_name}/pytorch_model.bin"    
    rm "${folder_name}/model.safetensors" 

    folder_name="exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-${formatted_number}"
    mkdir -p "$folder_name"    
    python magmax-ot.py --model1 exp/seq-fine/t5-small-wic --model2 exp/seq-fine/t5-small-wic-cb --model3 exp/seq-fine/t5-small-wic-cb-rte --model4 exp/seq-fine/t5-small-wic-cb-rte-copa --model5 exp/seq-fine/t5-small-wic-cb-rte-copa-boolq --model6 exp/seq-fine/t5-small-wic-cb-rte-copa-boolq-wsc --model7 exp/seq-fine/t5-small-wic-cb-rte-copa-boolq-wsc-multirc  --lamda $formatted_number --save_dir $folder_name

    python finetune.py --model_name_or_path $folder_name --do_predict --output_dir $folder_name  --per_device_eval_batch_size=32 --overwrite_output_dir  --predict_with_generate

    rm "${folder_name}/pytorch_model.bin"    
    rm "${folder_name}/model.safetensors"    
done
