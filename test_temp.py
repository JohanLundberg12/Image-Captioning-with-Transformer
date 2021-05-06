import sys
import pandas as pd

from configuration import Config
from load_data import extract_captions, extract_image_names, get_captions_file, load_captions

image_path = 'data/Flickr8k_Dataset/'
seed = 2

if __name__ == '__main__':
    lang = sys.argv[1]
    captions_file = get_captions_file(lang)

    df_train = pd.read_csv(f'data/{lang}_train.csv')
    df_val = pd.read_csv(f'data/{lang}_val.csv')
    df_test = pd.read_csv(f'data/{lang}_test.csv')

    captions_train = extract_captions(df_train)
    img_names_train = extract_image_names(df_train, image_path)

    captions_val = extract_captions(df_val)
    img_names_val = extract_image_names(df_val, image_path)

    captions_test = extract_captions(df_test)
    img_names_test = extract_image_names(df_test, image_path)
    
