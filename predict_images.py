# Standard Modules
import sys
import tensorflow as tf
from os import listdir

# Images
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

font = ImageFont.truetype('./arial.ttf', size=20)

# Custom Modules
from evaluation import evaluate
from helper_functions import get_checkpoint_path, load_config
from preprocessing import remove_unk
from pretrained_transformers import get_embedding, get_tokenizer
from transformer import build_transformer, CustomSchedule

config_file = sys.argv[1]
image_dir = sys.argv[2]
config = load_config(config_file)
config.checkpoint_path = get_checkpoint_path(config) #could be set in config file

tokenizer = get_tokenizer(config.lang)
embedding = get_embedding(config.embedding_type, config, tokenizer)
transformer = build_transformer(config, embedding, tokenizer)

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

images = [image_dir + image for image in listdir(image_dir) if image.split('.')[-1] == 'jpg']

for image in images:
    caption, result, attention_weights = evaluate(image, tokenizer, transformer)
    caption = remove_unk(caption, tokenizer.unk_token)
    caption = " ".join(caption)

    img = Image.open(image)
    img = img.resize((700, 375))
    width, height = img.size
    img2 = Image.new('RGB', (width,int(height+(height/5))), 'white')
    img2.paste(img)
    draw = ImageDraw.Draw(img2)
    draw.text((5, 400),caption,(0,0,0), font=font)

    image = image.split('/')[-1]
    img2.save(f'./data/extra_test_set/predictions/{config.lang}/{config.embedding_type}/{image}')