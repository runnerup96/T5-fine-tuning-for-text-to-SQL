import sys
import os
import re
import sqlite3
import json
import pickle
from tqdm import tqdm
import numpy as np

# path to https://github.com/taoyds/test-suite-sql-eval
sys.path.append("/sql_testing_suite")

import evaluation

sys.path.append("/T5-fine-tuning-for-text-to-SQL")

from data import processing
from data import compound_patching
from sklearn.model_selection import train_test_split
import uncertainty_data_path_consts as data_constants

EXECUTION_CACHE=dict()


def create_split(scores_matrix, target_matrix):
    seed_number = range(scores_matrix.shape[1])
    dev_scores_matrix, test_scores_matrix = [], []
    dev_target_matrix, test_target_matrix = [], []
    for i in seed_number:

        dev_scores, test_scores, dev_target, test_target = train_test_split(scores_matrix[:, i], target_matrix[:, i], test_size=0.66, random_state=42)
        dev_scores_matrix.append(dev_scores)
        test_scores_matrix.append(test_scores)
        
        dev_target_matrix.append(dev_target)
        test_target_matrix.append(test_target)
    
    dev_scores_matrix = np.array(dev_scores_matrix).T
    dev_target_matrix = np.array(dev_target_matrix).T
    
    test_scores_matrix = np.array(test_scores_matrix).T
    test_target_matrix = np.array(test_target_matrix).T
    
    return dev_scores_matrix, dev_target_matrix, test_scores_matrix, test_target_matrix

def create_exec_match_dict(test_list, preds_dict, split_name):
    exec_per_query = dict()
    for sample in tqdm(test_list):
        
        sample_id = sample['id']
        gold_query = sample['sql']
        pred_query = preds_dict[sample_id]['sql']
        
        cache_string = f"{gold_query}|{pred_query}|{split_name}"
        if cache_string not in EXECUTION_CACHE:
            if split_name != 'ehrsql':
                db_id = sample['db_id']
                db_path = os.path.join(data_constants.PAUQ_DB_PATH, db_id, db_id + ".sqlite")
                exec_match_result = eval_exec_match(gold_query, pred_query, db_path)
            else:
                exec_match_result = eval_ehrsql_match(gold_query, pred_query, data_constants.EHRSQL_MIMIC_PATH)
            
            EXECUTION_CACHE[cache_string] = exec_match_result
        else:
            exec_match_result = EXECUTION_CACHE[cache_string]
                
        if exec_match_result == 1:
            result = 0
        else:
            result = 1
        exec_per_query[sample_id] = result
        
    return exec_per_query


def read_gold_dataset_test(split_name, split_gold_path_dict):
    split_path = split_gold_path_dict[split_name]
    split = json.load(open(split_path, 'r'))
    split_list = []
    if split_name != 'ehrsql':
        for sample in split:
            new_sample = {"id":sample['id'], 
                      "sql":sample['query'], 
                      "db_id": sample['db_id']}
            split_list.append(new_sample)
    else:
        for key in split:
            new_sample = {"id": key, 
                      "sql": split[key], 
                      "db_id": 'mimic_iv'}
            split_list.append(new_sample)
        
    return split_list

def read_preds_dataset_test(split_name, model_name, seed, split_prediction_path_dict):
    preds = None
    model_preds_dict = split_prediction_path_dict.get(model_name)
    if model_preds_dict:
        split_pred_path = model_preds_dict.get(split_name)
        if split_pred_path:
            prediction_path = split_pred_path.format(seed=seed)
            preds = pickle.load(open(prediction_path, 'rb'))
    return preds

def make_scores_array(splits_file, preds_dict):
    scores_list = []
    for sample in splits_file:
        sample_id = sample['id']
        prediction_score = preds_dict[sample_id]['score']
        scores_list.append(prediction_score)
    return scores_list

def make_execution_result_array(splits_file, prediction_dict, split_name):
    execution_list = []
    exec_result_dict = create_exec_match_dict(splits_file, prediction_dict, split_name)
    for sample in splits_file:
        sample_id = sample['id']
        execution_status = exec_result_dict[sample_id]
        execution_list.append(execution_status)
    return execution_list

def make_numpy_arrays(split_name, model_name, seed_list, split_prediction_path_dict, split_gold_path_dict):
    split_test = read_gold_dataset_test(split_name, split_gold_path_dict)
    awailable_splits = seed_list[model_name]
    
    prediction_scores_matrix = []
    execution_result_matrix = []
    for seed in awailable_splits:
        prediction_file = read_preds_dataset_test(split_name, model_name, seed, split_prediction_path_dict)
        if prediction_file:
            scores_array = make_scores_array(split_test, prediction_file)
            prediction_scores_matrix.append(scores_array)

            execution_array = make_execution_result_array(split_test, prediction_file, split_name)
            execution_result_matrix.append(execution_array)
    
    if len(prediction_scores_matrix) != 0:
        prediction_scores_matrix = np.array(prediction_scores_matrix).T
        execution_result_matrix = np.array(execution_result_matrix).T

        print(prediction_scores_matrix.shape, execution_result_matrix.shape)
        return prediction_scores_matrix, execution_result_matrix
    else:
        print(f'No prediction for {model_name} for {split_name}!')
        return None, None


def extract_compounds(query, compound_max_len=4):
    # extract compounds of len 4(questionable - will check that later with compound grammar)
    return compound_patching.create_patch_list(query, compound_max_len = compound_max_len, extract_unique_compounds=False)


def parse_sql(query, db_path):
    schema = evaluation.Schema(evaluation.get_schema(db_path))
    try:
        parsed_query = evaluation.get_sql(schema, query)
    except:
        parsed_query = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
        }
    return parsed_query

def eval_exec_match(gold_query, pred_query, db_path):
    return evaluation.eval_exec_match(db=db_path, p_str=pred_query, g_str=gold_query, 
                                      plug_value=False, keep_distinct=False, 
                                      progress_bar_for_each_datapoint=False,
                                      run_async=False)


def eval_ehrsql_match(gold_query, pred_query, db_path):
    
    __current_time = "2100-12-31 23:59:00"
    __precomputed_dict = {
                    'temperature': (35.5, 38.1),
                    'sao2': (95.0, 100.0),
                    'heart rate': (60.0, 100.0),
                    'respiration': (12.0, 18.0),
                    'systolic bp': (90.0, 120.0),
                    'diastolic bp':(60.0, 90.0),
                    'mean bp': (60.0, 110.0)
                                }
    
    
    def post_process_sql(query):

        query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()
        query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')

        if "current_time" in query:
            query = query.replace("current_time", f"'{__current_time}'")
        if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
            vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
            vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
            vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
            if len(vital_name_list)==1:
                processed_vital_name = vital_name_list[0].replace('_', ' ')
                if processed_vital_name in __precomputed_dict:
                    vital_range = __precomputed_dict[processed_vital_name]
                    query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")

        query = query.replace("%y", "%Y").replace('%j', '%J')

        return query
    
    def process_answer(ans):
        return str(sorted([str(ret) for ret in ans[:100]])) # check only up to 100th record

    def execute(sql, db_path, skip_indicator='null'):
        if sql != skip_indicator:
            con = sqlite3.connect(db_path)
            con.text_factory = lambda b: b.decode(errors="ignore")
            cur = con.cursor()
            result = cur.execute(sql).fetchall()
            con.close()
            return process_answer(result)
        else:
            return skip_indicator
    
    def execute_query(sql1, sql2, db_path):
        try:
            result1 = execute(sql1, db_path)
        except:
            result1 = 'error1'
        try:
            result2 = execute(sql2, db_path)
        except:
            result2 = 'error2'
        result = {'real': result1, 'pred': result2}
        return result
    
    
    gold_query = post_process_sql(gold_query)
    pred_query = post_process_sql(pred_query)
    query_result = execute_query(gold_query, pred_query, db_path)
    
    exec_result = (query_result['real'] == query_result['pred'])
    return exec_result
    
    
    

def parse_complex_nested_dict_values(d):
    service_values = ['table_unit', 'and', 'or', 'desc', 'asc']
    if isinstance(d, dict):
        for value in d.values():
            if isinstance(value, dict) or isinstance(value, list) or isinstance(value, tuple):
                yield from parse_complex_nested_dict_values(value)
            elif value not in service_values:
                yield value
    elif isinstance(d, list) or isinstance(d, tuple):
        for value in d:
            if isinstance(value, dict) or isinstance(value, list) or isinstance(value, tuple):
                yield from parse_complex_nested_dict_values(value)
            elif value not in service_values:
                yield value
    elif value not in service_values:
        yield d
        
def filter_list(l):
    extracted_value = []
    for t in l:
        if isinstance(t, bool):
            continue
        elif isinstance(t, str):
            extracted_value.append(t)
        elif isinstance(t, float):
            extracted_value.append(str(t))
    return extracted_value

def get_sql_variables(query, db_path):
    """
    extracts attributes, tables, values from query
    """
    sql_dict = parse_sql(query, db_path)
    dict_values = parse_complex_nested_dict_values(sql_dict)
    extracted_vals = filter_list(dict_values)
    return extracted_vals
    
    
    