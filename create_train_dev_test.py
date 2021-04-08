import sys

from configuration import Config
from load_data import extract_captions, extract_image_names, get_captions_file, load_captions

image_path = 'data/Flickr8k_Dataset/'
seed = 2

if __name__ == '__main__':
    lang = sys.argv[1] #english or danish
    captions_file = get_captions_file(lang)

    df = load_captions(captions_file)
    df = df.sample(frac=1, random_state=seed).copy()

    train_pct = int(0.8*len(df))
    test_pct = int(0.9*len(df))
    train, dev, test = df[:train_pct], df[train_pct:test_pct], df[test_pct:]

    train.to_csv(f'data/{lang}_train.csv', index=False)
    dev.to_csv(f'data/{lang}_val.csv', index=False)
    test.to_csv(f'data/{lang}_test.csv', index=False)