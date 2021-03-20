# Standard Modules
import time
import tensorflow as tf

# Custom Modules
from load_data import *
from image_preprocessing import extract_image_features
from pretrained_transformers import get_danish_transformers, get_english_transformers
from transformer import *

# Loading data
lang = 'english'
image_path = 'data/Flickr8k_Dataset/'
captions_file = get_captions_file(lang)
seed = 42

df = load_captions(captions_file)
df = sample_from_df(df, 64, seed)

# Extracting captions and image names from the dataframe
all_captions = extract_captions(df)
all_image_names = extract_image_names(df, image_path)

# Getting image features
extract_image_features = False
if extract_image_features:
    extract_image_features(image_path, all_image_names)

# BertTokenizer & BertModel
if lang == 'danish':
    tokenizer, BertModel = get_danish_transformers()
elif lang =='english':
    tokenizer, BertModel = get_english_transformers()
else:
    print("Wrong language specified: ", lang)


input_ids = tokenizer(all_captions,
                      padding=True,
                      return_tensors="tf",
                      return_token_type_ids=False,
                      return_attention_mask=False)['input_ids']

# Splits
train_pct = int(0.8*len(all_image_names))
test_pct = int(0.9*len(all_image_names))
img_names_train, img_names_val, img_names_test = all_image_names[
    :train_pct], all_image_names[train_pct:test_pct], all_image_names[test_pct:]
cap_train, cap_val, cap_test = input_ids[:train_pct], input_ids[
    train_pct:test_pct], input_ids[test_pct:]

assert (len(img_names_train) == len(cap_train)) & (len(
    img_names_test) == len(cap_test)) & (len(img_names_val) == len(cap_val))

# Convert data to tensors
train_dataset = data_to_tensors(img_names_train, cap_train)
val_dataset = data_to_tensors(img_names_val, cap_val)
test_dataset = data_to_tensors(img_names_test, cap_test)

# Training parameters
vocab_size = tokenizer.vocab_size #should perhaps be fixed size and not entire vocab?
num_layer = 4
d_model = 768
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
target_vocab_size = vocab_size + 1
dropout_rate = 0.1

# Loss and Optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

# Instantiate model
transformer = Transformer(num_layer, d_model, num_heads, dff, row_size, col_size,
                          target_vocab_size, BertModel, max_pos_encoding=target_vocab_size,
                          rate=dropout_rate)

#Checkpointing
checkpoint_path = "./checkpoints"
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
ckpt.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# Training 
@tf.function
def train_step(img_tensor, tar):
    tar_inp = tar[:, :-1]  # cuts last column off
    tar_real = tar[:, 1:]  # cuts first column off
    dec_mask = create_masks_decoder(tar_inp) # creates a tensor mask of (tar_inp.shape[0], 1, tar_inp.shape[1], tar_inp.shape[1])
    # E.g. 8 (tar_inp.shape[0] -> batch size) instances of a list of a 15x15 matrice
    with tf.GradientTape() as tape:  # backprop.GradientTape
        predictions, _ = transformer(
            inp=img_tensor, tar=tar_inp, training=True, look_ahead_mask=dec_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


EPOCHS = 1

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    # (img_tensor: image, tar: input sentence)
    for (batch, (img_tensor, tar)) in enumerate(train_dataset):
        train_step(img_tensor, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

transformer.save_weights("./saved_models/ckpt")
