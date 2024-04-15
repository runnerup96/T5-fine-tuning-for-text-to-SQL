from collections import Counter
import random
import re

def find_text_vales(query):
    pattern = r"\'[^']+\'"
    result = re.findall(pattern, query)
    return result


def unveil_compound(compound_tokens):
    '''
    ATTRIBUTE_3 = ATTRIBUTE_3 -> [ATTRIBUTE_3 =, = ATTRIBUTE_3, ATTRIBUTE_3 = ATTRIBUTE_3]
    where ATTRIBUTE_4 >= NUMERIC_VALUE_1'] -> [where ATTRIBUTE_4, ATTRIBUTE_4 >=,
                                               where ATTRIBUTE_4 >=,
                                               where ATTRIBUTE_4 >= NUMERIC_VALUE_1, ATTRIBUTE_4 >= NUMERIC_VALUE_1]
    order by count ( ATTRIBUTE_5 ) desc - []
    '''
    unveiled_compounds = []
    for start_idx in range(len(compound_tokens)):
        for end_idx in range(start_idx + 1, len(compound_tokens)):
            compound = compound_tokens[start_idx:end_idx + 1]
            compound_str = " ".join(compound)
            unveiled_compounds.append(compound_str)
    return unveiled_compounds


def create_patch_list(query_str, compound_max_len, extract_unique_compounds=True):

    text_values = find_text_vales(query_str)
    value_dict = dict()
    for idx, tv in enumerate(text_values):
        query_str = query_str.replace(tv, f'VALUE_{idx}', 1)
        value_dict[f'VALUE_{idx}'] = tv

    query_tokens = query_str.split()
    compound_max_len = min(compound_max_len, len(query_tokens))

    query_compounds = []
    for i in range(0, len(query_tokens), compound_max_len):
        sublist = query_tokens[i:i + compound_max_len]
        if sublist:
            unveiled_compounds_list = unveil_compound(sublist)
            for compound_str in unveiled_compounds_list:
                if len(value_dict) > 0:
                    for k, v in value_dict.items():
                        compound_str = compound_str.replace(k, v)
                query_compounds.append(compound_str)
    compound_counter = Counter(query_compounds)
    # we do filtering in order not to have dubious patching
    if extract_unique_compounds:
        query_patches = list({k: v for k, v in compound_counter.items() if v == 1}.keys())
    else:
        query_patches = list(compound_counter.keys())
    return query_patches

#
#
# def form_patches_dict(query, parser):
#     query_grammar_catches = parser.get_compounds(query)
#     unveiled_compound_dict = {key: [] for key in query_grammar_catches}
#     all_compounds_list = []
#     for key in query_grammar_catches:
#         for compound_str in query_grammar_catches[key]:
#             unveiled_compound_list = unveil_compound(compound_str)
#             unveiled_compound_dict[key] += unveiled_compound_list
#             all_compounds_list += unveiled_compound_list
#     compound_counter = Counter(all_compounds_list)
#     # we do filtering in order not to have dubious patching
#     unique_compound_counter = {k: v for k, v in compound_counter.items() if v[1] == 1}
#     return unique_compound_counter


def prepare_patch_data(query, patch_length):
    patch_list = create_patch_list(query, patch_length)
    random_compound = random.sample(patch_list, 1)[0]
    query_with_compound_mask = query.replace(random_compound, 'COMPOUND')
    random_compounds_tokens = random_compound.split()
    random.shuffle(random_compounds_tokens)
    input_compound_str = " ".join(random_compounds_tokens)
    return {
        "shuffled_compound": f"COMPOUND: {input_compound_str}",
        "masked_query": query_with_compound_mask
    }