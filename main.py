# Standard Modules
import numpy as np
import time
import tensorflow as tf

# Custom Modules
from configuration import Config
from load_data import build_dataset
from pretrained_transformers import get_pretrained_bert_transformer, get_danish_tokenizer, get_english_tokenizer
from transformer import *


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Training

#@tf.function
def train_step(img_tensor, tar):
    tar_inp = tar[:, :-1]  # cuts last column off
    tar_real = tar[:, 1:]  # cuts first column off
    # creates a tensor mask of (tar_inp.shape[0], 1, tar_inp.shape[1], tar_inp.shape[1])
    dec_mask = create_masks_decoder(tar_inp)
    # E.g. 8 (tar_inp.shape[0] -> batch size) instances of a list of a 15x15 matrice
    with tf.GradientTape() as tape:  # backprop.GradientTape
        predictions, _ = transformer(
            inp=img_tensor, tar=tar_inp, training=True, look_ahead_mask=dec_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


if __name__ == '__main__':
    lang = 'english'
    embedding_type = 'pretrained' #pretrained or random
    config = Config(lang, embedding_type)
    seed = config.seed

    if config.lang == 'english':
        tokenizer = get_english_tokenizer()
    elif config.lang == 'danish':
        tokenizer = get_danish_tokenizer()
    else:
        raise NotImplementedError(f"{config.lang} not supported")

    if config.embedding_type == 'pretrained':
        model = get_pretrained_bert_transformer(config.lang)
        embedding = model.bert.embeddings.weights[0] #word_embeddings -> shape: 30522 x 768
    else:
        embedding = tf.keras.layers.Embedding(
            tokenizer.vocab_size, config.d_model)
    
    dataset_train, dataset_val, dataset_test = build_dataset(config, tokenizer)

    transformer = build_transformer(config, embedding, tokenizer)

    # Loss and Optimizer
    learning_rate = CustomSchedule(config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    
    # Checkpointing
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, config.checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    

    for epoch in range(config.epochs):
        print("Epoch: ", epoch)
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # (img_tensor: image, tar: input sentence)
        for (batch, (img_tensor, tar)) in enumerate(dataset_train):
            train_step(img_tensor, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    transformer.save_weights("./saved_models/ckpt")
