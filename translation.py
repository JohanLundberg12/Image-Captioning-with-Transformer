import os
import six
import time
from google.cloud import translate_v2 as translate
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/mnt/c/Users/johan/Data_Science/Bachelor/Image-Captioning-with-Transformer/My Project 55093-fccd3cabb42a.json"

def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)
    
    return result['translatedText']


def translate(caption):
    while True:
        try:
            translation = translate_text('da', caption)
        except Exception:
            time.sleep(100) #requests measured every 100 seconds. 
            continue
        break
    
    return translation


if __name__ == "__main__":
    file = 'data/Flickr8k_text/Flickr8k.token.txt'
    with open("data/Flickr8k_text/Flickr8k_danish.token.txt", "w") as da_captions_file:
        with open(file) as f:
            text = f.read()
            for line in text.split("\n"):
                if line:
                    col = line.split("\t")
                    caption_identifier = col[0].split("#") #image_name#id
                    caption = col[1].lower()
                    caption = translate(caption)
                    image_name = caption_identifier[0]
                    caption_id = caption_identifier[1]
                    da_captions_file.write(image_name + '#' + caption_id + "\t" + caption + "\n")