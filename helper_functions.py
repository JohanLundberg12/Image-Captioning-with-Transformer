import numpy as np

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