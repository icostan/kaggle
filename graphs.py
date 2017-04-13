import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")


def plot_correlation_map(df):
    _, ax = plt.subplots(figsize=(15, 10))
    _ = sns.heatmap(
        df.corr(),
        cmap="RdYlGn",
        square=True,
        cbar=True, cbar_kws={'shrink': .8},
        ax=ax,
        annot=True, annot_kws={'fontsize': 12})


def plot_accuracy(history):
    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')


def plot_loss(history):
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
