GOLD_TEST_SPLIT_PATH = {
    "pauq_xsp": "/pauq/pauq_xsp_test.json",
    "template_ssp": "/my_cp_splits/template_ssp_test.json",
    "tsl_ssp": "/my_cp_splits/tsl_ssp_test.json",
    "ehrsql": "/data/mimic_iv/test/label.json"
}

SPLITS_PREDICTIONS_PATH = {'t5-large':{
    "pauq_xsp": "/experiments/t5-large_pauq_xsp_s{seed}/pauq_xsp_test_inference_result.pkl",
    "template_ssp": "/experiments/t5-large_template_ssp_s{seed}/template_ssp_test_inference_result.pkl",
    "tsl_ssp": "/experiments/t5-large_tsl_ssp_s{seed}/tsl_ssp_test_inference_result.pkl",
    "ehrsql": "/ehrsql-text2sql-solution_statics/training_trials/t5-large_s{seed}/ehrsql_test_for_t5_inference_result.pkl"
},
                          't5-3b':{
    "pauq_xsp": "/experiments/t5-3b_pauq_xsp_s{seed}/pauq_xsp_test_inference_result.pkl",
    "template_ssp": "/experiments/t5-3b_template_ssp_s{seed}/template_ssp_test_inference_result.pkl",
    "tsl_ssp": "/experiments/t5-3b_tsl_ssp_s{seed}/tsl_ssp_test_inference_result.pkl",
    "ehrsql": "/ehrsql-text2sql-solution_statics/training_trials/t5-3b_s{seed}/ehrsql_test_for_t5_inference_result.pkl"
},
                          'dailsql':{
    "pauq_xsp": "/DAILSQL/dailsql_pauq_xsp_s{seed}_predicted_dict.pkl",
    "template_ssp": "/DAILSQL/dailsql_template_ssp_s{seed}_predicted_dict.pkl",
    "tsl_ssp": "/DAILSQL/dailsql_tsl_ssp_s{seed}_predicted_dict.pkl",
    "ehrsql": "/DAILSQL/dailsql_ehrsql_s{seed}_predicted_dict.pkl",
                          
},
                          "llama3_lora":{
    "pauq_xsp": "/text2sql_llama_3/experiments/pauq_pauq_xsp_s{seed}_lora/pauq_xsp_test_inference_result.pkl",
    "template_ssp": "/text2sql_llama_3/experiments/pauq_template_ssp_s{seed}_lora/template_ssp_test_inference_result.pkl",
    "tsl_ssp": "/text2sql_llama_3/experiments/pauq_tsl_ssp_s{seed}_lora/tsl_ssp_test_inference_result.pkl",
    "ehrsql": "/text2sql_llama_3/experiments/ehrsql_s{seed}_lora_more_epochs/test_inference_result.pkl"
},
                          "llama3_sft":{
    "pauq_xsp": "/text2sql_llama_3/experiments/pauq_pauq_xsp_s{seed}_sft_1_epoch/pauq_xsp_test_inference_result.pkl",
    "template_ssp": "/text2sql_llama_3/experiments/pauq_template_ssp_s{seed}_sft_1_epoch/template_ssp_test_inference_result.pkl",
    "tsl_ssp": "/text2sql_llama_3/experiments/pauq_tsl_ssp_s{seed}_sft_1_epoch/tsl_ssp_test_inference_result.pkl",
    "ehrsql": "/text2sql_llama_3/experiments/ehrsql_s{seed}_sft_3_epoch/test_inference_result.pkl"
}, 
}

SEED_LIST = {
    "t5-large": [1,42,123],
    "t5-3b": [1,42, 123],
    "llama3_lora": [1,42, 123],
    "dailsql": [1],
    "llama3_sft":[1, 42, 123]
}

MODEL_NAMES = {
    "t5-large": "T5-large",
    "t5-3b": "T5-3B",
    "llama3_lora": "Llama3-8B LoRA",
    "dailsql": "DIAL-SQL",
    "llama3_sft": "Llama3-8B SFT"
}

SPLITS_NAMES = {'pauq_xsp': "PAUQ XSP", 
                'template_ssp': "Template SSP split", 
                'tsl_ssp':"TSL SSP split", 
                "ehrsql": "EHRSQL"}


dataset_names = ['pauq', 'ehrsql']
models_name = ['t5-large', 't5-3b', 'dailsql', 'llama3_lora', 'llama3_sft']

PAUQ_DB_PATH = "/pauq/pauq_databases"
EHRSQL_MIMIC_PATH = "/ehrsql-text2sql-solution_statics/data/mimic_iv/mimic_iv.sqlite"