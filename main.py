# Standard Modules
import numpy as np
import sys
import time
import tensorflow as tf

# Custom Modules
from load_data import build_dataset
from helper_functions import *
from plots import *
from pretrained_transformers import get_embedding, get_tokenizer
from transformer import *


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


# Training

#@tf.function #for static execution
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
    

#@tf.function
def test_step(img_tensor, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    dec_mask = create_masks_decoder(tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp=img_tensor, tar=tar_inp, training=False, look_ahead_mask=dec_mask)

        loss = loss_function(tar_real, predictions)

    val_loss(loss)
    val_accuracy(tar_real, predictions)


if __name__ == '__main__':

    config_file = sys.argv[1]
    config = load_config(config_file)

    lang = config.lang
    embedding_type = config.embedding_type  # pretrained or random
    samples = config.samples #number or None for entire dataset
    seed = config.seed #not used? should set tf.set_random_seed()
    config.checkpoint_path = get_checkpoint_path(config) #could be set in config file

    tokenizer = get_tokenizer(lang)
    embedding = get_embedding(embedding_type, config, tokenizer)
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

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(
        name='val_accuracy')

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

    epoch_train_loss = []
    epoch_val_loss = []

    start = time.time()
    for epoch in range(config.epochs):
        print("Epoch: ", epoch)
        epoch_start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        train_loss_results = []
        train_accuracy_results = []
        val_loss_results = []
        val_accuracy_results = []

        print("Training on train set.")
        for (batch, (img_tensor, tar)) in enumerate(dataset_train): 
        # (img_tensor: image, tar: input sentence)
            train_step(img_tensor, tar)

            if batch % 4 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                train_loss_results.append(train_loss.result())
                train_accuracy_results.append(train_accuracy.result())
        plot_loss_and_accuracy(
            train_loss_results, train_accuracy_results, epoch+1, name='train', x_name='batches')

        print("Evaluating on the validation set.")
        for (batch, (img_tensor, tar)) in enumerate(dataset_val):
            test_step(img_tensor, tar)

            if batch % 4 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, val_loss.result(), val_accuracy.result()))
                val_loss_results.append(val_loss.result())
                val_accuracy_results.append(val_accuracy.result())

        plot_loss_and_accuracy(
            val_loss_results, val_accuracy_results, epoch+1, name='val', x_name='batches')

        epoch_train_loss.append(train_loss.result())
        epoch_val_loss.append(val_loss.result())

        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print('Epoch {} Loss {:.4f} Accuracy {:.4f} val loss {} val acc {}\n'.format(epoch + 1,
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result(),
                                                                                     val_loss.result(),
                                                                                     val_accuracy.result()))
        print(f'Time taken for 1 epoch: {time.time() - epoch_start:.2f} secs\n')

        if epoch + 1 % 2 == 0:
            stop_early = callback_early_stopping(
                epoch_train_loss, min_delta=0.1)
            if stop_early:
                print("Callback Early Stopping Signal Received")
                print("Terminating Training")
                break

    plot_loss_epochs(
        epoch_train_loss, epoch_val_loss, config.embedding_type, lang
    )
    print("Training Time: %s seconds" % (time.time() - start))
    transformer.save_weights("./saved_models/ckpt")
