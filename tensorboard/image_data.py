from __future__ import absolute_import, division, print_function
from datetime import datetime
import io
import itertools
from packaging import version
from six.moves import range
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

print("TensorFlow version: ", tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Shape: ", train_images[0].shape)
print("Label: ", train_labels[0], "->", class_names[train_labels[0]])

# img = np.reshape(train_images[0], (-1, 28, 28, 1))
# # print("Shape: ", img)
#
# logdir = "G:\\AI\\tensorboard\\logs\\image\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
# file_writer = tf.summary.create_file_writer(logdir)
#
# with file_writer.as_default():
#   images = np.reshape(train_images[0:25], (-1, 28, 28, 1))
#   tf.summary.image("25 training data examples", images, max_outputs=25, step=0)


# logdir = "G:\\AI\\tensorboard\\logs\\image\\plot\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
# file_writer = tf.summary.create_file_writer(logdir)
#
# def plot_to_image(figure):
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close(figure)
#     buf.seek(0)
#     image = tf.image.decode_png(buf.getvalue(), channels=4)
#     image = tf.expand_dims(image, 0)
#     return image
#
# def image_grid():
#     figure = plt.figure(figsize=(10, 10))
#     for i in range(25):
#         plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(train_images[i], cmap=plt.cm.binary)
#     return figure
#
# figure = image_grid()
# with file_writer.as_default():
#     tf.summary.image("Training data", plot_to_image(figure), step=0)




def plot_to_image(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  image = tf.expand_dims(image, 0)
  return image

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

logdir = "G:\\AI\\tensorboard\\logs\\image\\new\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

def log_confusion_matrix(epoch, logs):
    test_pred_raw = model.predict(test_images)
    test_pred = np.argmax(test_pred_raw, axis=1)
    cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

model.fit(
    train_images,
    train_labels,
    epochs=20,
    verbose=0, # Suppress chatty output
    callbacks=[tensorboard_callback, cm_callback],
    validation_data=(test_images, test_labels),
)



