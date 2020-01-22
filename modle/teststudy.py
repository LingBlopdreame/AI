from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pathlib
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
import sklearn.datasets as skd



(x, y) = np.transpose(np.loadtxt("data2.csv", dtype=float, delimiter=',', unpack=True))
z = tf.data.Dataset.from_tensor_slices((x, y))
# z = z.shuffle(1000)

# plt.plot(x1, y1, 'ro')
# plt.plot(x, y, 'ro')
# plt.show()

W, B = tf.Variable(tf.random.normal([22], stddev=0.1)), tf.Variable(tf.zeros([22]))

losses = []
pes = []

for epoch in range(100):
    for step, (X, Y) in enumerate(z):
        with tf.GradientTape() as tape:
            out = X @ W + B
            loss = tf.square(Y - out)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w, b])
        for p, g in zip([W, B], grads):
            p.assign_sub(lr * g)
            pes.append(p)
        losses.append(loss)
        if step == 21:
            print(epoch, step, loss)




# plt.plot(x, y)
# plt.show()

# print(x)
# print(y)
















# x, y = skd.make_moons(n_samples = 5000, noise = 0.2, random_state = 100)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#
# # plt.plot(x_train, y_train, 'ro')
# # plt.show()
#
#
# def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None,dark=False):
#     if (dark):
#         plt.style.use('dark_background')
#     else:
#         sns.set_style("whitegrid")
#     plt.figure(figsize=(16, 12))
#     axes = plt.gca()
#     axes.set(xlabel="$x_1$", ylabel="$x_2$")
#     plt.title(plot_name, fontsize=30)
#     plt.subplots_adjust(left=0.20)
#     plt.subplots_adjust(right=0.80)
#     if (XX is not None and YY is not None and preds is not None):
#         plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1,cmap=cm.Spectral)
#         plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5],cmap="Greys", vmin=0, vmax=.6)
#     # 绘制散点图，根据标签区分颜色
#     plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral,edgecolors='none')
#
# # 调用 make_plot 函数绘制数据的分布，其中 X 为 2D 坐标，y 为标签
# make_plot(x, y, "Classification Dataset Visualization ")
# plt.show()
#

# df = pd.DataFrame({ 'num_legs': [2, 4, 8, 0],
#                     'num_wings': [2, 0, 0, 0],
#                     'num_specimen_seen': [10, 2, 1, 8]},
#                     index=['falcon', 'dog', 'spider', 'fish'])
#
# df_train = df.sample(frac=0.8, random_state=0)
# df_test = df.drop(df_train.index)
#
# df_train_status = df_train.describe()
# df_train_status = df_train_status.transpose()

# print(df_train_status)

# def norm(x):
#     return (x - df_train_status['mean'])/df_train_status['std']
#
# df_train_norm = norm(df_train)
# df_train = tf.data.Dataset.from_tensor_slices(df_train_norm.values)
#
# tf.build
# y2 = tf.constant([1,3])
# y2 = tf.one_hot(y2, depth=10)
# y2 = tf.keras.losses.categorical_crossentropy(y2, x, from_logits=False)
# loss = tf.reduce_mean(y2)
# print(loss)


# plt.plot(x, y)
# plt.show()