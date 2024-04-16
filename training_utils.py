import pandas as pd
from transformers import EvalPrediction, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import EvalLoopOutput
import os
import re


def generated_query_simple_processor(query):
    query = query.split('|')[-1]
    query = query.strip()
    query = query.replace('< -', '<-')
    return query


def original_query_simple_processor(query):
    query = query.split('|')[-1]
    query = query.strip()
    return query


def model_post_processing_function(examples: list, outputs: EvalLoopOutput, tokenizer):
    # Decode the predicted tokens.
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    predictions = [generated_query_simple_processor(pred) for pred in decoded_preds]
    raw_references = [original_query_simple_processor(sample['target']) for sample in examples]

    return EvalPrediction(predictions=predictions, label_ids=raw_references)


def read_t5_tsv_dataset(dataset_path, tokenizer, input_max_length, output_max_length):
    samples = []
    df = pd.read_csv(dataset_path, sep='\t', header=None, keep_default_na=False, na_values=['NaN'])

    if df.shape[1] == 3:
        df.columns = ['id', 'source', 'target']
    elif df.shape[1] == 2:
        df.columns = ['id', 'source']
        df['target'] = None

    for id_, source, target in zip(df['id'].to_list(), df['source'].tolist(), df['target'].tolist()):
        formatted_source = 'semanticparse: ' + source
        source_tokens = tokenizer.encode(formatted_source, add_special_tokens=False, truncation=True,
                                         max_length=input_max_length)

        if target:
            formatted_target = target
            target_tokens = tokenizer.encode(formatted_target, add_special_tokens=False, truncation=True,
                                             max_length=output_max_length)
        else:
            target_tokens = None

        samples.append({'id': id_,
                        'source_tokens': source_tokens,
                        'target_tokens': target_tokens,
                        'source': source,
                        'target': target})
    return samples


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class TrainingStopCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self, steps):
        self.total_training_steps = steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.total_training_steps:
            control.should_training_stop = True


