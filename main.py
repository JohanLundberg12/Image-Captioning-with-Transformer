# Standard Modules
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# Transforme Module
from transformers import BertTokenizer, TFBertModel

# Custom Modules
from load_data import *
from image_preprocessing import extract_image_features
from transformer import *

image_path = 'data/Flickr8k_Dataset/'
captions_file = "data/Flickr8k_text/Flickr8k.token.txt"
state = 42
extract_image_features = False

# Loading data
df = load_captions(captions_file)
df = sample_from_df(df, 64)

# Extracting captions and image names from the dataframe
all_captions = extract_captions(df)
all_image_names = extract_image_names(df, image_path)

# Getting image features
if extract_image_features:
    extract_image_features(image_path, all_image_names)

# BertTokenizer & BertModel
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

BertModel = TFBertModel.from_pretrained(
    'bert-base-uncased', output_hidden_states=True)

batch = tokenizer(all_captions,
                  padding=True,
                  return_tensors="tf",
                  return_token_type_ids=False,
                  return_attention_mask=False)
outputs = BertModel(batch)
last_hidden_states = outputs.last_hidden_state

# Splits
train_pct = int(0.8*len(all_image_names))
test_pct = int(0.9*len(all_image_names))
img_names_train, img_names_val, img_names_test = all_image_names[
    :train_pct], all_image_names[train_pct:test_pct], all_image_names[test_pct:]
cap_train, cap_val, cap_test = last_hidden_states[:train_pct], last_hidden_states[
    train_pct:test_pct], last_hidden_states[test_pct:]

assert (len(img_names_train) == len(cap_train)) & (len(
    img_names_test) == len(cap_test)) & (len(img_names_val) == len(cap_val))

# Convert data to tensors(works same as a dataloader I believe)
train_dataset = data_to_tensors(img_names_train, cap_train)
val_dataset = data_to_tensors(img_names_val, cap_val)
test_dataset = data_to_tensors(img_names_test, cap_test)


# Training parameters
top_k = tokenizer.vocab_size
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
target_vocab_size = top_k + 1
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
transformer = Transformer(num_layer,d_model,num_heads,dff,row_size,col_size,
                          target_vocab_size, max_pos_encoding=target_vocab_size,
                          rate=dropout_rate)

@tf.function
def train_step(img_tensor, tar):
    print("\nTrain Step...")
    tar_inp = tar[:, :-1] #cuts last column off
    print("tar_input in train_step: ", tar_inp.shape)
    tar_real = tar[:, 1:] #cuts first column off
    print("tar_real in train_step: ", tar_real.shape)
    dec_mask = create_masks_decoder(tar_inp) #creates a tensor mask of (tar_inp.shape[0], 1, tar_inp.shape[1], tar_inp.shape[1])
                                            #E.g. 8(tar_inp.shape[0] -> batch size) instances of a list of a 15x15 matrice 
    print("combined mask (dec_mask): ", dec_mask.shape)
    with tf.GradientTape() as tape: #backprop.GradientTape
        print("-----------Calling Transformer with inputs and target--------------------------------------------------------")
        print("input img_tensor shape: ", img_tensor.shape)
        print("target tar_inp shape: ", tar_inp.shape)
        print("dec_mask shape: ", dec_mask.shape)
        
        predictions, _ = transformer(inp=img_tensor, tar=tar_inp, training=True, look_ahead_mask=dec_mask)
        
        
        
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)   
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


for epoch in range(1):
    print("Epoch: ", epoch)
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    print("train_loss: ", train_loss.result())
    print("train_accuracy: ", train_accuracy.result(), "\n")
    for (batch, (img_tensor, tar)) in enumerate(train_dataset): #(img_tensor: image, tar: input sentence)
        print("For loop...")
        print("batch: ", batch, "img_tensor shape: ", img_tensor.shape, "target shape: ", tar.shape)
        print("target: ", tar, "\n")
        train_step(img_tensor, tar)
        print()
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                               train_loss.result(),
                                               train_accuracy.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))