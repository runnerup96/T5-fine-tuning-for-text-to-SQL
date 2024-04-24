import json
import os.path
import argparse
import sys
import compound_patching
import random
import numpy as np

from tqdm import tqdm
import processing
from sql_metadata import Parser
import collections


max_compound_dict = {
        "random_ssp": 52,
        "paraphrase_ssp": 14,
        "trl_ssp": 49,
        "tsl_ssp": 44,
        "template_ssp": 39,
        "pauq_xsp": 44,
        "spider_xsp": 44,
        "shaw_spider_template_ssp": 39,
        "shaw_spider_length_ssp": 44,
        "shaw_spider_tmcd_ssp": 39,
        "shaw_spider_random_ssp": 52
}

def _get_schema_string(db_table_json):
    """Returns the schema serialized as a string."""
    table_id_to_column_names = collections.defaultdict(list)
    for table_id, name in db_table_json["column_names_original"]:
        table_id_to_column_names[table_id].append(name.lower())
    tables = db_table_json["table_names_original"]

    table_strings = []
    for table_id, table_name in enumerate(tables):
        column_names = table_id_to_column_names[table_id]
        table_string = " | %s : %s" % (table_name.lower(), " , ".join(column_names))
        table_strings.append(table_string)
    result_string = "".join(table_strings).lower().replace('\t', "")
    return result_string

def get_query_relevant_schema_string(query_tables, query_columns, db_table_json):
    table_id_to_column_names = collections.defaultdict(list)
    for table_id, name in db_table_json["column_names_original"]:
        table_id_to_column_names[table_id].append(name.lower())
    tables = db_table_json["table_names_original"]

    table_strings = []
    for table_id, table_name in enumerate(tables):
        if table_name.lower() in query_tables:
            column_names = table_id_to_column_names[table_id]
            relevant_column_names = list(filter(lambda x: x in query_columns, column_names))
            table_string = " | %s : %s" % (table_name.lower(), " , ".join(relevant_column_names))
            table_strings.append(table_string)
    result_string = "".join(table_strings).lower().replace('\t', "")
    return result_string


def prepare_sl_examples(examples, db2schema_str, dbid2schema_info, schema_linking):
    # мы ничего не будем выкидывать из инпута - просто будем обучать нашу модель предсказывать все схемы/таблицы для вопроса

    prepared_examples = []
    for idx, sample in tqdm(enumerate(examples), total=len(examples)):
        id_ = sample.get('id', str(idx))
        db_id = sample['db_id']
        # run parsing through processed shit
        question = processing.process_input_question(sample['question'])

        query = sample['query']
        processed_query = processing.normalize_sql_query(query)
        parsed_query = Parser(processed_query)
        query_tables, query_columns = parsed_query.tables, parsed_query.columns
        query_columns = [col.split('.')[-1] for col in query_columns]

        input_schema_string = db2schema_str[db_id]
        target_schema_string = get_query_relevant_schema_string(query_tables, query_columns, dbid2schema_info[db_id])

        if schema_linking:
            source = f"{db_id}: {question} {input_schema_string}"
            target = f"{db_id} {target_schema_string}"
        else:
            source = f"{db_id}: {question} {input_schema_string}"
            target = f"{db_id} | {processed_query}"

        prepared_examples.append((id_, source, target))
    return prepared_examples


def prepare_cp_examples(examples, dbid2schema_str, split_name, patch_samples=False):
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

def form_sl_dataset(examples, db_id_to_schema_string, db_id_to_schema_content, split_name,
                    schema_linking, phase, data_split, save_path):
    prepared_pt_train_examples = prepare_sl_examples(examples=examples,
                                                     db2schema_str=db_id_to_schema_string,
                                                     dbid2schema_info=db_id_to_schema_content,
                                                     schema_linking=schema_linking)
    if data_split == 'train' and split_name == 'spider_xsp':
        del prepared_pt_train_examples[3153]

    if phase in ['pt', 'ft']:
        filename = f"{phase}_{split_name}_ptr{pretrain_ratio_str}_{data_split}_schema_linking.tsv"
    else:
        filename = f"{split_name}_{data_split}.tsv"

    write_tsv(prepared_pt_train_examples, os.path.join(save_path, filename), expected_num_columns=3)

def form_cp_dataset(examples, db_id_to_schema_string, split_name, apply_pathing, phase, data_split, save_path):
    prepared_pt_train_examples = prepare_cp_examples(examples=examples,
                                                     dbid2schema_str=db_id_to_schema_string,
                                                     split_name=split_name,
                                                     patch_samples=apply_pathing)
    if phase in ['pt', 'ft']:
        filename = f"{phase}_{split_name}_ptr{pretrain_ratio_str}_{data_split}_comp_gen.tsv"
    else:
        filename = f"{split_name}_{data_split}.tsv"

    write_tsv(prepared_pt_train_examples, os.path.join(save_path, filename), expected_num_columns=3)


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

    parser = argparse.ArgumentParser('The testing components of')

    parser.add_argument('--splits_directory', default="spider", type=str)
    parser.add_argument('--seed', default=42, type=int, help='')
    parser.add_argument('--split_name', default="spider_xsp", type=str)
    parser.add_argument('--cp_pretrain', action='store_true')
    parser.add_argument('--sl_pretrain', action='store_true')
    parser.add_argument('--pretrain_ratio', default=1.0, type=float)
    args = parser.parse_args(sys.argv[1:])


    splits_dir = args.splits_directory
    tables_path = f"raw_splits/{splits_dir}/tables.json"
    tables_json = json.load(open(tables_path, 'r'))

    db_id_to_schema_string = {}
    db_id_to_schema_content = {}
    for table_json in tables_json:
        db_id = table_json["db_id"]
        db_id_to_schema_string[db_id] = _get_schema_string(table_json)
        db_id_to_schema_content[db_id] = table_json

    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    split_name = args.split_name
    make_cp_pretrain = args.cp_pretrain
    make_sl_pretrain = args.sl_pretrain

    pretrain_ratio = args.pretrain_ratio
    pretrain_ratio_str = str(pretrain_ratio).replace('.', '')
    split_dir_path = f"prepared_data/{split_name}"
    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    train_file_path = f"raw_splits/{splits_dir}/{split_name}_train.json"
    train_examples = json.load(open(train_file_path, 'r'))
    test_file_path = f"raw_splits/{splits_dir}/{split_name}_test.json"
    test_examples = json.load(open(test_file_path, 'r'))

    if make_cp_pretrain or make_sl_pretrain:
        # split data in pt_train, ft_train, pt_test, ft_test
        # random.shuffle(train_examples)
        pt_samples = int(pretrain_ratio * len(train_examples))
        if pt_samples == len(train_examples):
            pt_train_examples, ft_train_examples = train_examples, train_examples
        else:
            pt_train_examples, ft_train_examples = train_examples[:pt_samples], train_examples[pt_samples:]

        if make_cp_pretrain:

            form_cp_dataset(examples=pt_train_examples, db_id_to_schema_string=db_id_to_schema_string,
                            split_name=split_name, apply_pathing=True, phase='pt', data_split='train',
                            save_path=split_dir_path)

            form_cp_dataset(examples=ft_train_examples, db_id_to_schema_string=db_id_to_schema_string,
                            split_name=split_name, apply_pathing=False, phase='ft', data_split='train',
                            save_path=split_dir_path)

            form_cp_dataset(examples=test_examples, db_id_to_schema_string=db_id_to_schema_string,
                            split_name=split_name, apply_pathing=True, phase='pt', data_split='test',
                            save_path=split_dir_path)

            form_cp_dataset(examples=test_examples, db_id_to_schema_string=db_id_to_schema_string,
                            split_name=split_name, apply_pathing=False, phase='ft', data_split='test',
                            save_path=split_dir_path)

        elif make_sl_pretrain:
            form_sl_dataset(examples=pt_train_examples, db_id_to_schema_string=db_id_to_schema_string,
                            db_id_to_schema_content=db_id_to_schema_content, split_name=split_name,
                            schema_linking=True, phase='pt', data_split='train', save_path=split_dir_path)

            form_sl_dataset(examples=ft_train_examples, db_id_to_schema_string=db_id_to_schema_string,
                            db_id_to_schema_content=db_id_to_schema_content, split_name=split_name,
                            schema_linking=False, phase='ft', data_split='train', save_path=split_dir_path)

            form_sl_dataset(examples=test_examples, db_id_to_schema_string=db_id_to_schema_string,
                            db_id_to_schema_content=db_id_to_schema_content, split_name=split_name,
                            schema_linking=True, phase='pt', data_split='test', save_path=split_dir_path)

            form_sl_dataset(examples=test_examples, db_id_to_schema_string=db_id_to_schema_string,
                            db_id_to_schema_content=db_id_to_schema_content, split_name=split_name,
                            schema_linking=False, phase='ft', data_split='test', save_path=split_dir_path)

    else:
        # just run classic

        form_cp_dataset(examples=train_examples, db_id_to_schema_string=db_id_to_schema_string,
                        split_name=split_name, apply_pathing=False, phase='original', data_split='train',
                        save_path=split_dir_path)

        form_cp_dataset(examples=test_examples, db_id_to_schema_string=db_id_to_schema_string,
                        split_name=split_name, apply_pathing=False, phase='original', data_split='test',
                        save_path=split_dir_path)




    # TODO: Для pretraining делаем аналогичный валидационный тест
    # TODO: На претрейне мы меряем способность модели просто верно переставлять
    # TODO: На finetune меряем - способность модели генерить верный запрос
    # TODO: То есть у нас за 1 прогон обучения решается 1 задача

    # TODO: Но мне надо часть данных сплита оставить на finetune

    # Запрос в гугле - how to implement schedualed training?
