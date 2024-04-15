

import json

if __name__ == "__main__":
    split_name = "shaw_spider_random_ssp"
    json_path = f'raw_splits/shaw_splits/{split_name}_test.json'
    test_sample = json.load(open(json_path, "r"))

    with open(f"raw_splits/shaw_splits/{split_name}_test_gold.txt", "w") as f:
        for sample in test_sample:
            query, db_id = sample['query'], sample['db_id']
            query = query.replace('\t', '')
            gold_str = f"{query}\t{db_id}\n"
            f.write(gold_str)