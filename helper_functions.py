import numpy as np

from configuration import Config

def check_pretrained_embedding(pretrained_embedding, trained_embedding):
    if pretrained_embedding == trained_embedding:
        return "Training has not changed the original pretrained embeddings."
    else:
        return "Pretrained embeddings has been changed during training."


def callback_early_stopping(loss, min_delta):
    loss_prev = loss[-2:-1]
    loss_new = loss[:-1]

    delta_abs = np.abs(loss_new - loss_prev)
    delta_abs = np.abs(delta_abs / loss_prev)

    if delta_abs < min_delta:
        print("Loss didn't change much from last epoch")
        print("Percent change in loss value: ", delta_abs*1e2)
        return True
    else:
        return False


def get_checkpoint_path(config):
    return config.checkpoint_path + f'/{config.lang}/{config.embedding_type}'


def load_config(config_file):
    with open(config_file) as f:
        lines = f.read().split("\n")
        lang = lines[0].split(' ')[1]
        embedding_type = lines[1].split(' ')[1]
        samples = lines[2].split(' ')[1]

        if samples:
            samples = int(samples) #else None
    
    config = Config(lang, embedding_type, samples)

    return config



if __name__ == '__main__':
    load_config('./config_files/danish_random_debug.txt')