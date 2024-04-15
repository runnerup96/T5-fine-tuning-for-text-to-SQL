#!/bin/bash

#set -e
export BASE_DIR="/home/somov/naacl_cp_t5"
export SPLIT_NAME='random_ssp'
epoch=468
cp_mode="yes"
pretrain_ratio=$(echo "0.25" | bc)

# epoch num per split
# random_ssp - 468 / 29
# template_ssp - 520 / 32
# paraphrase_ssp - 988 / 61
# tsl_ssp - 520 / 32
# pauq_xsp - 206

CUDA_DEVICE_NUMBER='0'
seed='1'

train_batch_size=16
gradient_accumulation_steps=16
eval_batch_size=16

lr='1e-4'
data_dir="/home/somov/naacl_cp_t5/data/prepared_data"
save_model_dir="experiments"
model_name="t5-base"
dir_model_name="t5-base"

log_ratio=$(echo "0.1" | bc)
eval_ratio=$(echo "0.2" | bc)



if [ "$cp_mode" = "yes" ];
then
  pretrain_ratio_str="0${pretrain_ratio/./}"
  pt_train_file="$data_dir/$SPLIT_NAME/pt_${SPLIT_NAME}_ptr${pretrain_ratio_str}_train.tsv"
  pt_test_file="$data_dir/$SPLIT_NAME/pt_${SPLIT_NAME}_ptr${pretrain_ratio_str}_test.tsv"

  ft_train_file="$data_dir/$SPLIT_NAME/ft_${SPLIT_NAME}_ptr${pretrain_ratio_str}_train.tsv"
  ft_test_file="$data_dir/$SPLIT_NAME/ft_${SPLIT_NAME}_ptr${pretrain_ratio_str}_test.tsv"

  pt_epochs=$(echo "$epoch * $pretrain_ratio" | bc)
  #after pretraining we skip the pt epochs and
  ft_epochs_no_skip=$(echo "$epoch * ( 1 - $pretrain_ratio)" | bc)
  ft_epochs_with_skip=$(echo "$pt_epochs + $epoch * ( 1 - $pretrain_ratio)" | bc)

  pt_epochs=${pt_epochs%.*}
  ft_epochs_no_skip=${ft_epochs_no_skip%.*}
  ft_epochs_with_skip=${ft_epochs_with_skip%.*}

  pt_log_steps=$(echo "$pt_epochs * $log_ratio" | bc)
  pt_log_steps=${pt_log_steps%.*}
  pt_eval_steps=$(echo "$pt_epochs * $eval_ratio" | bc)
  pt_eval_steps=${pt_eval_steps%.*}

  ft_log_steps=$(echo "$ft_epochs_no_skip * $log_ratio" | bc)
  ft_log_steps=${ft_log_steps%.*}
  ft_eval_steps=$(echo "$ft_epochs_no_skip * $eval_ratio" | bc)
  ft_eval_steps=${ft_eval_steps%.*}

  pt_lr='1e-3'
  ft_lr='1e-4'

  run_name="${dir_model_name}_cp_${SPLIT_NAME}_s${seed}_ptr${pretrain_ratio_str}"
  output_dir="$save_model_dir/$run_name"
else
  train_file="$data_dir/$SPLIT_NAME/${SPLIT_NAME}_train.tsv"
  test_file="$data_dir/$SPLIT_NAME/${SPLIT_NAME}_test.tsv"

  run_name="${dir_model_name}_${SPLIT_NAME}_s$seed"
  output_dir="$save_model_dir/$run_name"

  log_steps=$(echo "$epoch * $log_ratio" | bc)
  log_steps=${log_steps%.*}
  eval_steps=$(echo "$epoch * $eval_ratio" | bc)
  eval_steps=${eval_steps%.*}
fi
logs_dir="$output_dir/logs"


tmux new-session -d -s $run_name
# compostionally pretrain
if [ "$cp_mode" = "yes" ];
then
  # first run pretraining phase
#  tmux send-keys -t $run_name "set -e" ENTER

  tmux send-keys -t $run_name "echo 'Pretraining epochs: $pt_epochs'" ENTER
  tmux send-keys -t $run_name "echo 'Finetuning epochs: $ft_epochs_no_skip $ft_epochs_with_skip'" ENTER

#  tmux send-keys -t $run_name "echo 'Run first pretraining stage...'" ENTER
#  tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u fine_tune_t5.py \
#                              --model_name_or_path $model_name \
#                              --train_file $pt_train_file \
#                              --validation_file $pt_test_file \
#                              --do_train \
#                              --prediction_loss_only \
#                              --learning_rate $pt_lr \
#                              --max_grad_norm 1.0 \
#                              --seed $seed \
#                              --per_device_train_batch_size $train_batch_size \
#                              --per_device_eval_batch_size $eval_batch_size \
#                              --gradient_accumulation_steps $gradient_accumulation_steps \
#                              --num_train_epochs $pt_epochs \
#                              --max_seq_length 512  \
#                              --max_output_length 256 \
#                              --generation_max_length 256 \
#                              --save_strategy 'steps' \
#                              --evaluation_strategy 'steps' \
#                              --eval_delay $pt_eval_steps \
#                              --eval_steps $pt_eval_steps \
#                              --save_steps $pt_eval_steps \
#                              --eval_accumulation_steps $gradient_accumulation_steps \
#                              --logging_steps $pt_log_steps \
#                              --report_to 'tensorboard' \
#                              --save_total_limit 1 \
#                              --overwrite_output_dir \
#                              --output_dir '$output_dir/pretrain' \
#                              --logging_dir '$logs_dir/pretrain' \
#                              --phase 'pretrain'" ENTER
  tmux send-keys -t $run_name "echo 'Run second finetuning stage'" ENTER
  # then run finetuning phase
  tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u fine_tune_t5.py \
                              --model_name_or_path '$output_dir/pretrain' \
                              --train_file $ft_train_file \
                              --validation_file $ft_test_file \
                              --do_train \
                              --do_eval \
                              --predict_with_generate \
                              --learning_rate $ft_lr \
                              --max_grad_norm 1.0 \
                              --seed $seed \
                              --per_device_train_batch_size $train_batch_size \
                              --per_device_eval_batch_size $eval_batch_size \
                              --gradient_accumulation_steps $gradient_accumulation_steps \
                              --num_train_epochs $ft_epochs_with_skip \
                              --max_seq_length 512  \
                              --max_output_length 256 \
                              --generation_max_length 256 \
                              --save_strategy 'steps' \
                              --evaluation_strategy 'steps' \
                              --eval_delay $ft_eval_steps \
                              --eval_steps $ft_eval_steps \
                              --save_steps $ft_eval_steps \
                              --eval_accumulation_steps $gradient_accumulation_steps \
                              --num_beams 1 \
                              --logging_steps $ft_log_steps \
                              --report_to 'tensorboard' \
                              --save_total_limit 1 \
                              --overwrite_output_dir \
                              --output_dir '$output_dir/finetune' \
                              --logging_dir '$logs_dir/finetune' \
                              --phase 'finetune'" ENTER
else
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
                              --logging_dir $logs_dir \
                              --phase 'original'" ENTER
fi
# finetune
tmux a -t $run_name
