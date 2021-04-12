# Standard Modules
import numpy as np
import sys
import tensorflow as tf

# Image Modules
import matplotlib.pyplot as plt
from PIL import Image

# Custom Modules
from evaluation import evaluate
from helper_functions import get_checkpoint_path, load_config
from preprocessing import remove_unk
from pretrained_transformers import get_embedding, get_tokenizer
from transformer import build_transformer, CustomSchedule

config_file = sys.argv[1]
image = sys.argv[2]
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
ckpt.restore(ckpt_manager.latest_checkpoint)

caption, result, attention_weights = evaluate(image, tokenizer, transformer)
caption = remove_unk(caption, tokenizer.unk_token)

temp_image = np.array(Image.open(image))
print("\nCaption: ", caption, "\n")
plt.imshow(temp_image)
plt.show()