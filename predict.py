import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
from transformers import BertTokenizer, TFBertModel

from image_preprocessing import load_image, load_inception_v3
from transformer import create_masks_decoder, Transformer

image = 'data/Flickr8k_Dataset/667626_18933d713e.jpg'

tokenizer = tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
BertModel = TFBertModel.from_pretrained(
        'bert-base-uncased', output_hidden_states=True)

vocab_size = tokenizer.vocab_size
num_layer = 4
d_model = 768
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
target_vocab_size = vocab_size + 1
dropout_rate = 0.1

transformer = Transformer(num_layer, d_model, num_heads, dff, row_size, col_size,
                          target_vocab_size, BertModel, max_pos_encoding=target_vocab_size,
                          rate=dropout_rate)
load_status = transformer.load_weights('saved_models/ckpt').expect_partial()
#load_status.assert_consumed()

image_features_extract_model = load_inception_v3()

def evaluate(image_path):
    temp_input = tf.expand_dims(load_image(image_path)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    start_token = '[CLS]'
    end_token = '[SEP]'
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0) #tokens
    result = [] #word list

    for i in range(100):
        dec_mask = create_masks_decoder(output)
        predictions, attention_weights = transformer(img_tensor_val,output,False,dec_mask)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        print(predictions)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if predicted_id == end_token:
            return result,tf.squeeze(output, axis=0), attention_weights
        result.append(tokenizer.index_word[int(predicted_id)])
        output = tf.concat([output, predicted_id], axis=-1)

    return result,tf.squeeze(output, axis=0), attention_weights

caption, result, attention_weights = evaluate(image)

for i in caption:
    if i=="<unk>":
        caption.remove(i)

result_join = ' '.join(caption)
result_final = result_join.rsplit(' ', 1)[0]

temp_image = np.array(Image.open(image))
plt.imshow(temp_image)