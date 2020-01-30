from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class mydense(tf.keras.layers.Layer):
    def __init__(self, output_num=32):
        super(mydense, self).__init__()
        self.output_num = output_num

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.output_num), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.output_num,), initializer='zeros', trainable=True)

    def call(self, inputs):
        out = tf.matmul(inputs, self.w) + self.b
        out = tf.nn.relu(out)
        return out

# x = tf.ones([3,4])
# n = mydense(2)
# y = n(x)
# print(y)

# (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test.shape)
# batchsz = 800
# train_db = tf.data.Dataset.from_tensor_slices((x, y))
# train_db = train_db.shuffle(1000)
# train_db = train_db.batch(batchsz)
# train_db = train_db.map(preprocess)
# train_db = train_db.repeat(20)

# test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
# x,y = next(iter(train_db))
# print('train sample:', x.shape, y.shape)

network = tf.keras.models.Sequential([mydense(512),
                                      mydense(256),
                                      mydense(128),
                                      mydense(64),
                                      mydense(32),
                                      mydense(10)])
network(tf.ones(shape=[3,5]))
network.summary()

# class Linear(tf.keras.layers.Layer):
#
#   def __init__(self, units=32):
#     super(Linear, self).__init__()
#     self.units = units
#
#   def build(self, input_shape):
#     self.w = self.add_weight(shape=(input_shape[-1], self.units),
#                              initializer='random_normal',
#                              trainable=True)
#     self.b = self.add_weight(shape=(self.units,),
#                              initializer='random_normal',
#                              trainable=True)
#
#   def call(self, inputs):
#     return tf.matmul(inputs, self.w) + self.b
#
# x = tf.ones((2, 2))
# linear_layer = Linear(32)  # At instantiation, we don't know on what inputs this is going to get called
# y = linear_layer(x)  # The layer's weights are created dynamically the first time the layer is called