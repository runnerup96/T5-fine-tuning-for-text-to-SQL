import collections
import json
import os.path

from tqdm import tqdm
import processing
import compound_patching
import random
import numpy as np


max_compound_dict = {
        "random_ssp": 52,
        "paraphrase_ssp": 14,
        "trl_ssp": 49,
        "tsl_ssp": 44,
        "template_ssp": 39,
        "pauq_xsp": 44,
        "shaw_spider_template_ssp": 39,
        "shaw_spider_length_ssp": 44,
        "shaw_spider_tmcd_ssp": 39,
        "shaw_spider_random_ssp": 52
}

def _get_schema_string(table_json):
    """Returns the schema serialized as a string."""
    table_id_to_column_names = collections.defaultdict(list)
    for table_id, name in table_json["column_names_original"]:
        table_id_to_column_names[table_id].append(name.lower())
    tables = table_json["table_names_original"]

    table_strings = []
    for table_id, table_name in enumerate(tables):
        column_names = table_id_to_column_names[table_id]
        table_string = " | %s : %s" % (table_name.lower(), " , ".join(column_names))
        table_strings.append(table_string)
    result_string = "".join(table_strings).lower().replace('\t', "")
    return result_string


def prepare_examples(examples, dbid2schema_str, split_name, patch_samples=False):
    prepared_examples = []
    for idx, sample in tqdm(enumerate(examples), total=len(examples)):
        id_ = sample.get('id', str(idx))
        db_id = sample['db_id']
        schema_str = dbid2schema_str[db_id]
        # run parsing through processed shit
        question = processing.process_input_question(sample['question'])

        query = sample['query']
        processed_query = processing.normalize_sql_query(query)

        if patch_samples:
            max_patch_length = max_compound_dict[split_name]
            patch_data = compound_patching.prepare_patch_data(processed_query, max_patch_length)
            source = f"{db_id}: {question} | {patch_data['shuffled_compound']} {schema_str}"
            masked_query = patch_data['masked_query']
            target = f"{db_id} | {masked_query}"
        else:
            source = f"{db_id}: {question} {schema_str}"
            target = f"{db_id} | {processed_query}"

        prepared_examples.append((id_, source, target))
    return prepared_examples


def write_tsv(examples, filename, expected_num_columns=2):
    """Write examples to tsv file."""
    with open(filename, "w") as tsv_file:
        for example in examples:
            if len(example) != expected_num_columns:
                raise ValueError("Example '%s' has %s columns." %
                                 (example, len(example)))
            example = "\t".join(example)
            line = "%s\n" % example
            tsv_file.write(line)
    print("Wrote %s examples to %s." % (len(examples), filename))


if __name__ == "__main__":
    splits_dir = "my_cp_splits"
    tables_path = f"raw_splits/{splits_dir}/tables.json"
    tables_json = json.load(open(tables_path, 'r'))

    db_id_to_schema_string = {}
    for table_json in tables_json:
        db_id = table_json["db_id"]
        db_id_to_schema_string[db_id] = _get_schema_string(table_json)

    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    split_name = "random_ssp"
    apply_pathing = True

    pretrain_ratio = 0.25
    pretrain_ratio_str = str(pretrain_ratio).replace('.', '')
    split_dir_path = f"prepared_data/{split_name}"
    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    train_file_path = f"raw_splits/{splits_dir}/{split_name}_train.json"
    train_examples = json.load(open(train_file_path, 'r'))
    test_file_path = f"raw_splits/{splits_dir}/{split_name}_test.json"
    test_examples = json.load(open(test_file_path, 'r'))

    if apply_pathing:
        #split data in pt_train, ft_train, pt_test, ft_test
        random.shuffle(train_examples)
        pt_samples = int(pretrain_ratio * len(train_examples))
        pt_train_examples, ft_train_examples = train_examples[:pt_samples], train_examples[pt_samples:]

        prepared_pt_train_examples = prepare_examples(examples=pt_train_examples,
                                   dbid2schema_str=db_id_to_schema_string,
                                   split_name=split_name,
                                   patch_samples=apply_pathing)
        filename = f"pt_{split_name}_ptr{pretrain_ratio_str}_train.tsv"
        write_tsv(prepared_pt_train_examples, os.path.join(split_dir_path, filename), expected_num_columns=3)

        prepared_ft_train_examples = prepare_examples(examples=ft_train_examples,
                                                      dbid2schema_str=db_id_to_schema_string,
                                                      split_name=split_name,
                                                      patch_samples=False)
        filename = f"ft_{split_name}_ptr{pretrain_ratio_str}_train.tsv"
        write_tsv(prepared_ft_train_examples, os.path.join(split_dir_path, filename), expected_num_columns=3)

        # for eval I want to evaluate on the same dataset
        # I see that the loss has dropped on dataset - now I run the second phase of training
        prepared_pt_test_examples = prepare_examples(examples=test_examples,
                                                  dbid2schema_str=db_id_to_schema_string,
                                                 split_name=split_name,
                                                 patch_samples=apply_pathing)
        filename = f"pt_{split_name}_ptr{pretrain_ratio_str}_test.tsv"
        write_tsv(prepared_pt_test_examples, os.path.join(split_dir_path, filename), expected_num_columns=3)

        prepared_ft_test_examples = prepare_examples(examples=test_examples,
                                                     dbid2schema_str=db_id_to_schema_string,
                                                     split_name=split_name,
                                                     patch_samples=False)
        filename = f"ft_{split_name}_ptr{pretrain_ratio_str}_test.tsv"
        write_tsv(prepared_ft_test_examples, os.path.join(split_dir_path, filename), expected_num_columns=3)

    else:
        # just run classic
        prepared_train_examples = prepare_examples(examples=train_examples,
                                                   dbid2schema_str=db_id_to_schema_string,
                                                   split_name=split_name,
                                                   patch_samples=apply_pathing)
        filename = f"{split_name}_train.tsv"
        write_tsv(prepared_train_examples, os.path.join(split_dir_path, filename), expected_num_columns=3)

        prepared_test_examples = prepare_examples(examples=test_examples,
                                                  dbid2schema_str=db_id_to_schema_string,
                                                  split_name=split_name,
                                                  patch_samples=apply_pathing)
        filename = f"{split_name}_test.tsv"
        write_tsv(prepared_test_examples, os.path.join(split_dir_path, filename), expected_num_columns=3)




    # TODO: Для pretraining делаем аналогичный валидационный тест
    # TODO: На претрейне мы меряем способность модели просто верно переставлять
    # TODO: На finetune меряем - способность модели генерить верный запрос
    # TODO: То есть у нас за 1 прогон обучения решается 1 задача

    # TODO: Но мне надо часть данных сплита оставить на finetune

    # Запрос в гугле - how to implement schedualed training?
