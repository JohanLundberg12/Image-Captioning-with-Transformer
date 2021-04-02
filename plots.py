import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def plot_loss_and_accuracy(loss, accuracy, epoch, lang, embedding_type, name):

    fig, axes = plt.subplots(2, sharex=True, figsize=(15,8))
    fig.suptitle(f"{name} Metrics")

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Batches", fontsize=14)
    axes[1].plot(accuracy)
    plt.savefig(f"./plots/{lang}/{lang}_{embedding_type}_{name}_loss_accuracy_epoch_{epoch}.jpg")
    
def plot_loss_epochs(epoch_train_loss, epoch_val_loss, embedding_type, lang):

    fig, axes = plt.subplots(2, sharex=True, figsize=(15,8))
    fig.suptitle("Epoch Train/Val Loss")

    axes[0].set_ylabel("Train Loss", fontsize=14)
    axes[0].plot(epoch_train_loss)

    axes[1].set_ylabel("Val Loss", fontsize=14)
    axes[1].set_xlabel("Epochs", fontsize=14)
    axes[1].plot(epoch_val_loss)
    plt.savefig(f"./plots/{lang}/{lang}_{embedding_type}_loss_epochs.jpg")