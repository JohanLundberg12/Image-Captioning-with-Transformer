import string

def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(string.punctuation)
    
    return(text_no_punctuation)

def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    
    return(text_len_more_than1)

def remove_numeric(text):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if isalpha:
            text_no_numeric += " " + word
    
    return(text_no_numeric)

def text_clean(text_original):
    text = remove_punctuation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)
    
    return(text)

def clean_df(df):
    for i, caption in enumerate(df.caption.values):
        newcaption = text_clean(caption)
        df["caption"].iloc[i] = newcaption

    return df


def remove_unk(caption, unk_token):
    for token in caption:
        if token == unk_token:
            caption.remove(token)
    
    return caption

