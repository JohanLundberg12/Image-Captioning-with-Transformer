import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu

from image_preprocessing import load_image, load_inception_v3
from transformer import create_masks_decoder


def evaluate(image_path, tokenizer, transformer):
    image_features_extract_model = load_inception_v3()
    
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
        result.append(tokenizer.convert_ids_to_tokens(int(predicted_id)))
        output = tf.concat([output, predicted_id], axis=-1)

    return result,tf.squeeze(output, axis=0), attention_weights



def bleu1(reference, candidate):
    return sentence_bleu([reference], candidate, weights=(1.0, 0, 0, 0)) * 100


def bleu2(reference, candidate):
    return sentence_bleu([reference], candidate, weights=(0.5, 0.5, 0, 0)) * 100


def bleu3(reference, candidate):
    return sentence_bleu([reference], candidate, weights=(0.3, 0.3, 0.3, 0)) * 100


def bleu4(reference, candidate):
    return sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25)) * 100

