import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import seaborn as sns


x = tf.random.normal([2,10])
print(x)
y = tf.constant([[1,3],[2,4]])
print(y)
y = tf.one_hot(y, depth=10)
print(y)

x2 = x.sampl


# y2 = tf.constant([1,3])
# y2 = tf.one_hot(y2, depth=10)
# y2 = tf.keras.losses.categorical_crossentropy(y2, x, from_logits=False)
# loss = tf.reduce_mean(y2)
# print(loss)


# plt.plot(x, y)
# plt.show()