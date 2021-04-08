# Standard Modules
import numpy as np
import tensorflow as tf
import sys

# Image Modules
import matplotlib.pyplot as plt
from PIL import Image

# Custom Modules
from helper_functions import get_checkpoint_path, load_config
from image_preprocessing import load_inception_v3, load_image
from pretrained_transformers import get_embedding, get_tokenizer
from transformer import build_transformer, create_masks_decoder, CustomSchedule

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

image_features_extract_model = load_inception_v3()

def evaluate(image_path):
    temp_input = tf.expand_dims(load_image(image_path)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0) #tokens
    result = [] #word list

    for i in range(100):
        dec_mask = create_masks_decoder(output)
        predictions, attention_weights = transformer(img_tensor_val,output,False,dec_mask)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if predicted_id == end_token:
            return result,tf.squeeze(output, axis=0), attention_weights
        result.append(tokenizer.ids_to_tokens[int(predicted_id)])
        output = tf.concat([output, predicted_id], axis=-1)

    return result,tf.squeeze(output, axis=0), attention_weights

caption, result, attention_weights = evaluate(image)

for i in caption:
    if i==tokenizer.unk_token:
        caption.remove(i)

result_join = ' '.join(caption)
result_final = result_join.rsplit(' ', 1)[0]

temp_image = np.array(Image.open(image))
print()
print("Caption: ", caption)
print()
plt.imshow(temp_image)
plt.show()