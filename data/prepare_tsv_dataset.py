import json
import os.path
import argparse
import sys
import random
import numpy as np

from tqdm import tqdm
import processing
from sql_metadata import Parser
import collections


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


def prepare_examples(examples, dbid2schema_str, split_name):
    prepared_examples = []
    for idx, sample in tqdm(enumerate(examples), total=len(examples)):
        id_ = sample.get('id', str(idx))
        db_id = sample['db_id']
        schema_str = dbid2schema_str[db_id]
        question = processing.process_input_question(sample['question'])

        query = sample['query']
        processed_query = processing.normalize_sql_query(query)

        source = f"{db_id}: {question} {schema_str}"
        target = f"{db_id} | {processed_query}"

        prepared_examples.append((id_, source, target))
    return prepared_examples


def form_dataset(examples, db_id_to_schema_string, split_name, data_split, save_path):
    prepared_pt_train_examples = prepare_examples(examples=examples,
                                                     dbid2schema_str=db_id_to_schema_string,
                                                     split_name=split_name)

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

    split_dir_path = f"prepared_data/{split_name}"
    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    train_file_path = f"raw_splits/{splits_dir}/{split_name}_train.json"
    train_examples = json.load(open(train_file_path, 'r'))
    test_file_path = f"raw_splits/{splits_dir}/{split_name}_test.json"
    test_examples = json.load(open(test_file_path, 'r'))


    form_dataset(examples=train_examples, db_id_to_schema_string=db_id_to_schema_string,
                    split_name=split_name, data_split='train',
                    save_path=split_dir_path)

    form_dataset(examples=test_examples, db_id_to_schema_string=db_id_to_schema_string,
                    split_name=split_name, data_split='test',
                    save_path=split_dir_path)