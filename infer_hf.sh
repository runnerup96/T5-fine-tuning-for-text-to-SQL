#!/bin/bash

#set -e
export BASE_DIR="/home/somov/naacl_cp_t5"
export SPLIT_NAME='dr_spider'

CUDA_DEVICE_NUMBER='0'
seed='3'

eval_batch_size=32
gradient_accumulation_steps=64

data_dir="/home/somov/naacl_cp_t5/data/prepared_data"

checkpoint_path="/home/somov/naacl_cp_t5/experiments/t5-3b_cp_pauq_xsp_s3_ptr025/finetune"
#test_file="$data_dir/$SPLIT_NAME/${SPLIT_NAME}_test.tsv"
test_file="/home/somov/naacl_cp_t5/data/prepared_data/dr_spider/dr_spider_all_samples.tsv"

run_name="dr_spider_3b_cp_infer"

tmux new-session -d -s $run_name

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u fine_tune_t5.py \
                              --model_name_or_path $checkpoint_path \
                              --validation_file $test_file \
                              --do_predict \
                              --predict_with_generate \
                              --seed $seed \
                              --per_device_eval_batch_size $eval_batch_size \
                              --max_seq_length 512  \
                              --max_output_length 256 \
                              --generation_max_length 256 \
                              --eval_accumulation_steps $gradient_accumulation_steps \
                              --num_beams 1 \
                              --output_dir $checkpoint_path" ENTER

tmux a -t $run_name
