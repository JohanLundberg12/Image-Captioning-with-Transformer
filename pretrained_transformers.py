
# Transforme Module
from transformers import BertTokenizer, TFBertModel #english
from transformers import AutoTokenizer, AutoModelForPreTraining #danish

def get_danish_transformers():
    tokenizer = AutoTokenizer.from_pretrained(
        'Maltehb/danish-bert-botxo', do_lower_case=True)
    BertModel = AutoModelForPreTraining.from_pretrained('Maltehb/danish-bert-botxo', output_hidden_states=True)

    return tokenizer, BertModel

def get_english_transformers():
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)

    BertModel = TFBertModel.from_pretrained(
        'bert-base-uncased', output_hidden_states=True)

    return tokenizer, BertModel