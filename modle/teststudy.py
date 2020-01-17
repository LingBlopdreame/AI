from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pathlib
import seaborn as sns

df = pd.DataFrame({ 'num_legs': [2, 4, 8, 0],
                    'num_wings': [2, 0, 0, 0],
                    'num_specimen_seen': [10, 2, 1, 8]},
                    index=['falcon', 'dog', 'spider', 'fish'])

df_train = df.sample(frac=0.8, random_state=0)
df_test = df.drop(df_train.index)

df_train_status = df_train.describe()
df_train_status = df_train_status.transpose()

print(df_train_status)

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