import pandas as pd
import numpy as np
import tensorflow as tf


def get_captions_file(lang):
    if lang == 'danish':
        captions_file = "data/Flickr8k_text/Flickr8k_danish.token.txt"
    elif lang == 'english':
        captions_file = "data/Flickr8k_text/Flickr8k.token.txt"

    return captions_file


def load_captions(caption_file):
    with open(caption_file) as file:
        text = file.read()
        data_txt = []
        for line in text.split("\n"):
            if line:
                col = line.split('\t')
                caption_identifier = col[0].split("#")  # image_name#id
                caption = col[1].lower()
                image_name = caption_identifier[0]
                caption_id = caption_identifier[1]
                data_txt.append([caption_id, image_name, caption])

    data = pd.DataFrame(data_txt, columns=["id", "image_name", "caption"])
    id_unique = np.unique(data.image_name.values)

    print("Total captions: {}".format(data.shape[0]))  # 40455
    print("Total unique image names: {}".format(len(id_unique)))  # 8091

    return data


def extract_captions(df):
    all_captions = []
    for caption in df['caption']:
        # possibly strip caption of empty space before first word and after last word
        caption = '[CLS] ' + caption + ' [SEP]'
        all_captions.append(caption)

    print(f"len(all_captions) : {len(all_captions)}")

    return all_captions


def extract_image_names(df, image_path):
    all_image_names = []
    for image_name in df['image_name']:
        full_image_path = image_path + image_name
        all_image_names.append(full_image_path)

    print(f"len(all_image_names) : {len(all_image_names)}")

    return all_image_names


def sample_from_df(df, num=None, seed=None):
    df = df.sample(frac=1, random_state=seed).copy()
    if num:
        df = df[:num]

    return df


def map_func(img_name, caption):
    # load image features .npy files
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, caption


def data_to_tensors(img_names, captions, config):
    batch_size = config.batch_size # size used during training
    buffer_size = 1000
    dataset = tf.data.Dataset.from_tensor_slices((img_names, captions))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [
                          tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def build_dataset(config, tokenizer):
    lang = config.lang
    image_path = config.image_path
    captions_file = get_captions_file(lang)
    df = load_captions(captions_file)
    df = sample_from_df(df, config.samples, config.seed)
    
    all_captions = extract_captions(df)
    all_image_names = extract_image_names(df, image_path)

    input_ids = tokenizer(all_captions,
                      padding=True,
                      return_tensors="tf",
                      return_token_type_ids=False,
                      return_attention_mask=False)['input_ids']
    # Splits
    train_pct = int(0.8*len(all_image_names))
    test_pct = int(0.9*len(all_image_names))
    img_names_train, img_names_val, img_names_test = all_image_names[
        :train_pct], all_image_names[train_pct:test_pct], all_image_names[test_pct:]
    cap_train, cap_val, cap_test = input_ids[:train_pct], input_ids[
        train_pct:test_pct], input_ids[test_pct:]
    dataset_train = data_to_tensors(img_names_train, cap_train, config)
    dataset_val = data_to_tensors(img_names_val, cap_val, config)
    dataset_test = data_to_tensors(img_names_test, cap_test, config)

    return dataset_train, dataset_val, dataset_test
