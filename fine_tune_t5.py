import logging
import os
import math

import torch
from transformers import HfArgumentParser, T5ForConditionalGeneration, AutoTokenizer, AutoConfig, Adafactor, \
    set_seed, Seq2SeqTrainingArguments, get_cosine_schedule_with_warmup
import evaluation
import hf_arguments
import text2sql_dataset
import training_utils
from sp_seq2seq_trainer import SemanticParsingSeq2SeqTrainer

def main():
    logger = logging.getLogger(__name__)

    hf_parser = HfArgumentParser((hf_arguments.ModelArguments, hf_arguments.DataTrainingArguments,
                                  Seq2SeqTrainingArguments, hf_arguments.ExperimentArgs))
    model_args, data_args, training_args, experiment_args = hf_parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_max_length=data_args.max_seq_length,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    tokenizer.add_tokens([" <", " <="])

    # Prepare model & optimizer

    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)

    train_samples, train_dataset = [], []
    optimizer, lr_scheduler = None, None
    if training_args.do_train:
        train_samples = training_utils.read_t5_tsv_dataset(data_args.train_file,
                                                       tokenizer=tokenizer,
                                                       input_max_length=data_args.max_seq_length,
                                                       output_max_length=data_args.max_output_length)


        if data_args.try_one_batch:
            one_batch_size = 32
            one_batch_samples = train_samples[-one_batch_size:]
            train_dataset = text2sql_dataset.T5FinetuneDataset(one_batch_samples, tokenizer)
            test_dataset = text2sql_dataset.T5FinetuneDataset(one_batch_samples, tokenizer)
        else:
            train_dataset = text2sql_dataset.T5FinetuneDataset(train_samples, tokenizer)

        optimizer = Adafactor(model.parameters(), lr=training_args.learning_rate,
                              scale_parameter=False, relative_step=False, clip_threshold=1.0,
                              warmup_init=False)
        # train step calculation

        batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = len(train_dataset) // batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        total_train_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)

        num_warmup_steps = int(0.1 * total_train_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                       num_training_steps=total_train_steps)
        print('My total train steps: ', total_train_steps)


    if training_args.do_eval or training_args.do_predict:
        test_samples = training_utils.read_t5_tsv_dataset(data_args.validation_file,
                                                      tokenizer=tokenizer,
                                                      input_max_length=data_args.max_seq_length,
                                                      output_max_length=data_args.max_output_length)
        test_dataset = text2sql_dataset.T5FinetuneDataset(test_samples, tokenizer)

    # prepare evaluation class
    evaluator = evaluation.Evaluator()

    trainer = SemanticParsingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        eval_examples=test_samples,
        tokenizer=tokenizer,
        compute_metrics=evaluator.evaluate,
        optimizers=(optimizer, lr_scheduler),
        post_process_function=training_utils.model_post_processing_function,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_state()

    if training_args.do_eval and experiment_args.phase != 'pretrain':
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=data_args.max_seq_length,
                                   num_beams=data_args.num_beams, metric_key_prefix="eval",
                                   output_save_dir=training_args.output_dir)

        if 'predictions' in metrics:
            output_dir = training_args.output_dir
            filename = os.path.basename(output_dir).split('.')[0]
            filename = f"{filename}_prediction.txt"
            save_path = os.path.join(output_dir, filename)
            with open(save_path, 'w') as f:
                for pred in metrics['predictions']:
                    f.write(f"{pred} \n")
            metrics.pop('predictions')

        metrics["eval_samples"] = len(test_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        res = trainer.predict(test_dataset, tokenizer=tokenizer)
        # Save the prediction files for spider evaluation
        prediction_list = []
        for pred_idx, pred_id in enumerate(res.predictions):
            prediction_list.append(pred_id)

        output_dir = training_args.output_dir
        filename = os.path.basename(data_args.validation_file).split('.')[0]
        filename = f"{filename}_prediction.txt"
        save_path = os.path.join(output_dir, filename)

        logger.info("Writing model predictions to txt file...")
        with open(save_path, 'w') as f:
            for line in prediction_list:
                f.write(f"{line}\n")


if __name__ == "__main__":
    main()
