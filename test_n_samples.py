import pandas as pd
import sys
import tensorflow as tf

from PIL import Image, ImageFont, ImageDraw

from evaluation import evaluate
from helper_functions import get_checkpoint_path, load_config
from load_data import extract_captions, extract_image_names, sample_from_df
from preprocessing import remove_unk
from pretrained_transformers import get_embedding, get_tokenizer
from transformer import build_transformer, CustomSchedule

seed = 2
font = ImageFont.truetype('./arial.ttf', size=20)

if __name__ == '__main__':
    config_file = sys.argv[1]
    n = int(sys.argv[2]) #should be 100
    config = load_config(config_file)
    config.checkpoint_path = get_checkpoint_path(config)
    
    tokenizer = get_tokenizer(config.lang)
    embedding = get_embedding(config.embedding_type, config, tokenizer)
    transformer = build_transformer(config, embedding, tokenizer)

    df_test = pd.read_csv(f'data/{config.lang}_test.csv')
    df_test = sample_from_df(df_test, n, seed)
    captions_test = extract_captions(df_test)
    img_names_test = extract_image_names(df_test, config.image_path)

    learning_rate = CustomSchedule(config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)

    # Loading Checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, config.checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    ckpt_path = tf.train.get_checkpoint_state(f'./checkpoints/{config.lang}/{config.embedding_type}/32/ckpt-1/').all_model_checkpoint_paths[14] #ckpt-15 best checkpoint
    ckpt.restore(ckpt_path)

    for image in img_names_test:
        caption, result, attention_weights = evaluate(image, tokenizer, transformer)
        caption = remove_unk(caption, tokenizer.unk_token)
        caption = ' '.join(caption)

        img = Image.open(image)
        img = img.resize((700, 375))
        width, height = img.size
        img2 = Image.new('RGB', (width,int(height+(height/5))), 'white')
        img2.paste(img)
        draw = ImageDraw.Draw(img2)
        draw.text((5, 400),caption,(0,0,0), font=font)

        image = image.split('/')[-1]
        img2.save(f'./data/10_percent_test_set/{config.lang}/{config.embedding_type}/{image}')
