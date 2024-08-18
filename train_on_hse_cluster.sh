#!/bin/bash
# запуск на кластере
# сначала актвиируем окружение - conda activate irm_env
# srun/sbatch -A proj_1406 train_on_hse_cluster.sh

# run config
#SBATCH --job-name=rand_3b   # Название задачи
#SBATCH --error=/home/t5/logs/rand_3b.err       # Файл для вывода ошибок
#SBATCH --output=/home/t5/logs/rand_3b.log       # Файл для вывода результатов
#SBATCH --time=36:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --gpus=1                            # Требуемое кол-во GPU

set -e
export BASE_DIR="/home/t5"
export SPLIT_NAME='random_ssp'
epoch=29
cp_mode="no"
pretrain_ratio=$(echo "0.25" | bc)

# epoch num per split
# random_ssp - 468 / 29
# template_ssp - 520 / 32
# paraphrase_ssp - 988 / 61
# tsl_ssp - 520 / 32
# pauq_xsp - 204

seed='42'

train_batch_size='4'
eval_batch_size='2'

lr='1e-4'
data_dir="/home/t5/data/prepared_data"
save_model_dir="/home/t5/experiments"
model_name="t5-3b"
dir_model_name="t5-3b"

if [ "$cp_mode" = "yes" ];
then
  pretrain_ratio_str="0${pretrain_ratio/./}"
  pt_train_file="$data_dir/$SPLIT_NAME/pt_${SPLIT_NAME}_ptr${pretrain_ratio_str}_train.tsv"
  pt_test_file="$data_dir/$SPLIT_NAME/pt_${SPLIT_NAME}_ptr${pretrain_ratio_str}_test.tsv"

  ft_train_file="$data_dir/$SPLIT_NAME/ft_${SPLIT_NAME}_ptr${pretrain_ratio_str}_train.tsv"
  ft_test_file="$data_dir/$SPLIT_NAME/ft_${SPLIT_NAME}_ptr${pretrain_ratio_str}_test.tsv"

  pt_epochs=$(echo "$epoch * $pretrain_ratio" | bc)
  #after pretraining we skip the pt epochs and
  ft_epochs=$(echo "$pt_epochs + $epoch * ( 1 - $pretrain_ratio)" | bc)

  pt_epochs=${pt_epochs%.*}
  ft_epochs=${ft_epochs%.*}

  pt_lr='1e-3'
  ft_lr='1e-4'

  run_name="${dir_model_name}_cp_${SPLIT_NAME}_s${seed}_ptr${pretrain_ratio_str}"
  output_dir="$save_model_dir/${dir_model_name}_cp_${SPLIT_NAME}_s${seed}_ptr${pretrain_ratio_str}"
else
  ft_train_file="$data_dir/$SPLIT_NAME/${SPLIT_NAME}_train.tsv"
  ft_test_file="$data_dir/$SPLIT_NAME/${SPLIT_NAME}_train.tsv"

  run_name="${dir_model_name}_${SPLIT_NAME}_s$seed"
  output_dir="$save_model_dir/${dir_model_name}_${SPLIT_NAME}_s$seed"
fi


# compostionally pretrain
if [ "$cp_mode" = "yes" ];
then
  # first run pretraining phase
  echo "Run first pretraining stage..."
  python -u fine_tune_t5.py \
          --model_name_or_path $model_name \
          --train_file $pt_train_file \
          --validation_file $pt_test_file \
          --do_train \
          --predict_with_generate \
          --seed $seed \
          --per_device_train_batch_size $train_batch_size \
          --per_device_eval_batch_size $eval_batch_size \
          --learning_rate $pt_lr \
          --num_train_epochs $pt_epochs \
          --gradient_accumulation_steps $train_batch_size \
          --max_seq_length 512  \
          --max_output_length 256 \
          --save_strategy 'epoch' \
          --evaluation_strategy 'steps' \
          --eval_steps 500 \
          --generation_max_length 256 \
          --save_total_limit 3 \
          --overwrite_output_dir \
          --output_dir $output_dir \
          --cp_mode \
          --phase 'pretrain'
  echo "Run second finetuning stage"
  # then run finetuning phase
  python -u fine_tune_t5.py \
        --model_name_or_path $model_name \
        --train_file $ft_train_file \
        --validation_file $ft_test_file \
        --ignore_data_skip \
        --do_train \
        --do_eval \
        --do_predict \
        --seed $seed \
        --predict_with_generate \
        --per_device_train_batch_size $train_batch_size \
        --per_device_eval_batch_size $eval_batch_size \
        --learning_rate $ft_lr \
        --num_train_epochs $ft_epochs \
        --gradient_accumulation_steps $train_batch_size \
        --max_seq_length 512  \
        --max_output_length 256 \
        --save_strategy 'epoch' \
        --metric_for_best_model 'exact_match' \
        --evaluation_strategy 'steps' \
        --eval_steps 250 \
        --generation_max_length 256 \
        --save_total_limit 3 \
        --output_dir $output_dir \
        --cp_mode \
        --phase 'finetune'

else
  python -u fine_tune_t5.py \
        --model_name_or_path $model_name \
        --train_file $ft_train_file \
        --validation_file $ft_test_file \
        --ignore_data_skip \
        --do_train \
        --do_eval \
        --do_predict \
        --seed $seed \
        --predict_with_generate \
        --per_device_train_batch_size $train_batch_size \
        --per_device_eval_batch_size $eval_batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --gradient_accumulation_steps $train_batch_size \
        --max_seq_length 512  \
        --max_output_length 256 \
        --save_strategy 'epoch' \
        --metric_for_best_model 'exact_match' \
        --evaluation_strategy 'steps' \
        --eval_steps 250 \
        --generation_max_length 256 \
        --save_total_limit 3 \
        --overwrite_output_dir \
        --output_dir $output_dir \
        --cp_mode False \
        --phase 'original'
fi
