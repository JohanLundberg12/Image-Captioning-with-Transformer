from helper_functions import load_config
import sys
import pandas as pd

from load_data import extract_captions, tokenize_captions
from helper_functions import load_config
from pretrained_transformers import get_tokenizer

config_file = sys.argv[1]

config = load_config(config_file)
tokenizer = get_tokenizer(config.lang)

df_train = pd.read_csv(f'data/{config.lang}_extra_test.csv')

captions_train = extract_captions(df_train)

ids_train = tokenize_captions(captions_train, tokenizer)

ids_train = ids_train.numpy()

words = {}

def add_to_dict(dic, arr):
    for row in arr:
        for token in row:
            if token not in dic:
                dic[token] = token
    return dic

words = add_to_dict(words, ids_train)
print(len(words))


    
