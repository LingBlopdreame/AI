import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

x, y = make_moons(n_samples=1000, noise=0.25, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None,dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.08, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, cmap=plt.cm.Spectral, edgecolors='none')

# make_plot(x, y, "Classification Dataset Visualization ")
# plt.show()

# for n in range(5):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Dense(8, input_dim=2, activation='relu'))
#     counter = 0
#     for _ in range(5):
#         model.add(tf.keras.layers.Dense(32, activation='relu'))
#         if counter < n:
#             counter += 1
#             model.add(tf.keras.layers.Dropout(rate=0.5))
#     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     log_dir="G:\\AI\\ch09_guonihe\\logs\\Dropout45\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#     history = model.fit(x_train,
#                         y_train,
#                         epochs=20,
#                         verbose=1,
#                         callbacks=[tensorboard_callback])
#     predss = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
#     title = "网络层数({})".format(n)
#     make_plot(x_train, y_train, title, XX=xx, YY=yy, preds=predss)
#     plt.show()

def build_model_with_regularization(_lambda):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_dim=2, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)))
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

k = [1e-10, 1e-3, 1e-1, 0.12, 0.13]
for _lambda in k:
    model = build_model_with_regularization(_lambda)
    log_dir="G:\\AI\\ch09_guonihe\\logs\\zhenze33\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(x_train,
                        y_train,
                        epochs=20,
                        verbose=1,
                        callbacks=[tensorboard_callback])
    predss = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
    title = "zhenzehua-[lambda = {}]".format(str(_lambda))
    make_plot(x_train, y_train, title, XX=xx, YY=yy, preds=predss)
    plt.show()
