import tensorflow as tf
import keras
import itertools
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt


# Function which apply the callbacks
def get_callbacks(name):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=20),
        tf.keras.callbacks.TensorBoard(log_dir=name, histogram_freq=1)
    ]


# Focal Loss used for imbalanced dataset
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


# Compile the model and fit
def compile_and_fit(model, train_data, valid_data,
                    name, class_weight=None, max_epochs=10):
    """
  Input:
    model: model to train
    train_data: dataset training
    valid_data dataset validation
    name: directory of tensorflow callbacks logs
  Output:
    history: return of fit function
  """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  # loss=focal_loss(),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.75),
                           tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           tf.keras.metrics.AUC(name='AUC'),
                           tf.keras.metrics.RecallAtPrecision(0.8, name='R_P80'),
                           tf.keras.metrics.RecallAtPrecision(0.9, name='R_P90'),
                           tf.keras.metrics.RecallAtPrecision(0.95, name='R_P95')])

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=max_epochs,
        callbacks=get_callbacks(name),
        shuffle=False,
        class_weight=class_weight,
        verbose=1)

    return history


def plot_confusion_matrix(cm, class_names):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
