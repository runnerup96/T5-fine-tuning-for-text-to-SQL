#!/bin/bash

export SPLIT_NAME='tsl_ssp'
epoch=520

# epoch num per split
# random_ssp - 468 / 29
# template_ssp - 520 / 32
# paraphrase_ssp - 988 / 61
# tsl_ssp - 520 / 32
# pauq_xsp - 206

CUDA_DEVICE_NUMBER='0'
seed='1'

train_batch_size=32
gradient_accumulation_steps=8
eval_batch_size=16

lr='1e-4'
# # Path to data files, in TSV format for T5 training
data_dir="/home/t5/data/prepared_data"
# Path, where experiments results will be stored
save_model_dir="experiments"
# (144) Path to HF model locally or remotely on HF
model_name="t5-base"
# Where model will be stored
dir_model_name="t5-base"

log_ratio=$(echo "0.1" | bc)
eval_ratio=$(echo "3.0" | bc)

log_steps=$(echo "$epoch * $log_ratio" | bc)
log_steps=${log_steps%.*}

eval_steps=$(echo "$epoch * $eval_ratio" | bc)
eval_steps=${eval_steps%.*}


train_file="$data_dir/$SPLIT_NAME/${SPLIT_NAME}_train.tsv"
test_file="$data_dir/$SPLIT_NAME/${SPLIT_NAME}_test.tsv"

run_name="${dir_model_name}_${SPLIT_NAME}_s$seed"
output_dir="$save_model_dir/$run_name"

logs_dir="$output_dir/logs"


tmux new-session -d -s $run_name

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u fine_tune_t5.py \
                            --model_name_or_path $model_name \
                            --train_file $train_file \
                            --validation_file $test_file \
                            --do_train \
                            --do_eval \
                            --predict_with_generate \
                            --learning_rate $lr \
                            --max_grad_norm 1.0 \
                            --seed $seed \
                            --per_device_train_batch_size $train_batch_size \
                            --per_device_eval_batch_size $eval_batch_size \
                            --gradient_accumulation_steps $gradient_accumulation_steps \
                            --num_train_epochs $epoch \
                            --max_seq_length 512  \
                            --max_output_length 256 \
                            --generation_max_length 256 \
                            --save_strategy 'steps' \
                            --evaluation_strategy 'steps' \
                            --metric_for_best_model 'eval_exact_match' \
                            --load_best_model_at_end \
                            --eval_delay $eval_steps \
                            --eval_steps $eval_steps \
                            --save_steps $eval_steps \
                            --eval_accumulation_steps $gradient_accumulation_steps \
                            --num_beams 1 \
                            --logging_steps $log_steps \
                            --report_to 'tensorboard' \
                            --save_total_limit 1 \
                            --overwrite_output_dir \
                            --output_dir $output_dir \
                            --logging_dir $logs_dir" ENTER


tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u infer_hf_t5.py \
                              --model_name_or_path $output_dir \
                              --test_file $test_file \
                              --seed $seed \
                              --max_seq_length 512 \
                              --max_output_length 256 \
                              --per_device_eval_batch_size $eval_batch_size \
                              --eval_accumulation_steps $gradient_accumulation_steps \
                              --generation_max_length 256 \
                              --num_beams 1 \
                              --output_dir $output_dir" ENTER

tmux a -t $run_name

