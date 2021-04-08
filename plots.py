import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

plt.rcParams['figure.figsize'] = [10, 10]
plt.style.use('fivethirtyeight')

def plot_loss_and_accuracy(loss, accuracy, epoch, lang, embedding_type, name):

    fig, axes = plt.subplots(2, sharex=True, figsize=(15,8))
    fig.suptitle(f"{name} Metrics")

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Batches", fontsize=14)
    axes[1].plot(accuracy)
    plt.savefig(f"./plots/{lang}/{embedding_type}/{lang}_{embedding_type}_{name}_loss_accuracy_epoch_{epoch}.jpg")
    plt.clf()


def plot_acc_epochs(epoch_train_acc, epoch_val_acc, embedding_type, lang):
    plt.plot(epoch_train_acc)
    plt.plot(epoch_val_acc)

    plt.title(f'{lang} {embedding_type} Accuracy')
    plt.ylabel("accuracy", fontsize=14)
    plt.xlabel("epoch", fontsize=14)
    plt.xticks(np.arange(1, len(epoch_train_acc)))
    plt.legend(["train", "val"], loc='upper right')
    plt.savefig(f"./plots/{lang}/{embedding_type}/{lang}_{embedding_type}_accuracy_epochs.jpg")
    plt.clf()

    
def plot_loss_epochs(epoch_train_loss, epoch_val_loss, embedding_type, lang):

    plt.plot(epoch_train_loss)
    plt.plot(epoch_val_loss)

    plt.title(f'{lang} {embedding_type} Loss')
    plt.ylabel("loss", fontsize=14)
    plt.xlabel("epoch", fontsize=14)
    plt.xticks(np.arange(1, len(epoch_train_loss)))
    plt.legend(['train', 'val'], loc='upper right')

    #plt.savefig(f"./plots/test.jpg")    
    plt.savefig(f"./plots/{lang}/{embedding_type}/{lang}_{embedding_type}_loss_epochs.jpg")
    plt.clf()


if __name__ == '__main__':
    pass
    #x = [0.5, 0.4, 0.35, 0.32, 0.31, 0.3, 0.29, 0.28, 0.28, 0.27, 999]
    #x2 = [0.7, 0.6, 0.5, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0]
    #y = [6, 5, 4, 3, 2, 1]
    #y2 = [10, 8, 7, 7, 8, 9]

    #plot_loss_epochs(x, x2, 'random', 'danish')
    #plot_loss_epochs(y, y2, 'random', 'danish')