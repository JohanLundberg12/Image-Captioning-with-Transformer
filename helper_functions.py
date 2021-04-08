import ast
import numpy as np
import sys

from configuration import Config

def check_pretrained_embedding(pretrained_embedding, trained_embedding):
    if pretrained_embedding == trained_embedding:
        return "Training has not changed the original pretrained embeddings."
    else:
        return "Pretrained embeddings has been changed during training."


def callback_early_stopping(loss, min_delta=0.1):
    loss_prev = loss[-2]
    loss_new = loss[-1]

    delta_abs = np.abs(loss_new - loss_prev)
    delta_abs = np.abs(delta_abs / loss_prev)

    if delta_abs < min_delta:
        print(f"Loss didn't change much from last epoch:\n Old loss: {loss_prev}\n New loss: {loss_new}")
        print(f"Percent change in loss value: {delta_abs*1e2} %")
        return True
    else:
        return False


def get_checkpoint_path(config):
    return config.checkpoint_path + f'/{config.lang}/{config.embedding_type}/{config.batch_size}/ckpt-1'


def load_config(config_file):
    with open(config_file) as f:
        lines = f.read().split("\n")
        lang = lines[0].split(' ')[1]
        embedding_type = lines[1].split(' ')[1]
        samples = lines[2].split(' ')[1]
         
    if samples != 'None':
        samples = int(samples)
    else:
        samples = None

    config = Config(lang, embedding_type, samples)

    return config


if __name__ == '__main__':
    f = sys.argv[1]
    load_config(f)