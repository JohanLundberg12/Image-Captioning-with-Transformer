import keras
import tensorflow as tf
from tensorflow.python.ops.variables import trainable_variables

# Transforme Module
from transformers import BertTokenizer, TFBertModel #english
from transformers import AutoTokenizer, AutoModelForPreTraining #danish

def get_danish_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
            'Maltehb/danish-bert-botxo', do_lower_case=True)

    return tokenizer


def get_danish_transformer():
    BertModel = AutoModelForPreTraining.from_pretrained('Maltehb/danish-bert-botxo', output_hidden_states=True)

    return BertModel


def get_english_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    
    return tokenizer

def get_english_transformer():
    BertModel = TFBertModel.from_pretrained(
        'bert-base-uncased', output_hidden_states=True)

    return BertModel

def get_pretrained_bert_transformer(lang):
    if lang == 'danish':
        return get_danish_transformer()
    elif lang == 'english':
        return get_english_transformer()
    else:
        raise NotImplementedError(f"{lang} not supported")


def get_tokenizer(lang):
    if lang == 'english':
        tokenizer = get_english_tokenizer()
    elif lang == 'danish':
        tokenizer = get_danish_tokenizer()
    else:
        raise NotImplementedError(f"{lang} not supported")

    return tokenizer


def get_embedding(embedding_type, config, tokenizer):
    if embedding_type == 'pretrained':
        model = get_pretrained_bert_transformer(config.lang) # word_embeddings -> shape: 30522 x 768
        embedding_matrix = model.bert.embeddings.weights[0]
        embedding_matrix = embedding_matrix.numpy()
        embedding = tf.keras.layers.Embedding(tokenizer.vocab_size, 
                                                config.d_model,
                                                trainable=False,
                                                weights=[embedding_matrix],
                                                input_shape=(None,))

    elif embedding_type == 'random':
        embedding = tf.keras.layers.Embedding(
            tokenizer.vocab_size, config.d_model)
    else:
        raise NotImplementedError(f"{embedding_type} not supprted")

    return embedding