import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=[28,28,1]),
#                              tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
#                              tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),])
# model.summary()

def f(y, out, m, n):
    plt.figure()
    for i in range(out):
        plt.subplot(m, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(y[:,:,i], cmap='gray')
    plt.show()


(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x1 = x_train[3]
plt.imshow(x1, cmap='gray')
plt.show()
print(x1.shape)
x = np.reshape(x1, [-1,28,28,1])
x = tf.cast(x, tf.float32)

convn1 = tf.keras.layers.Conv2D(32, [3,3], activation='relu', padding='SAME')
pooling = tf.keras.layers.AveragePooling2D([3,3], strides=2)
convn2 = tf.keras.layers.Conv2D(64, [3,3], activation='relu', padding='SAME')
convn3 = tf.keras.layers.Conv2D(64, [3,3], activation='relu', padding='SAME')
y11 = convn1(x)
y11 = pooling(y11)
y22 = convn2(y11)
y22 = pooling(y22)
y33 = convn3(y22)
y33 = pooling(y33)
# print(y.shape)
y1 = tf.squeeze(y11)
y2 = tf.squeeze(y22)
y3 = tf.squeeze(y33)
figure = plt.figure()
# plt.subplot(4, 8, 1)
# plt.subplot(4, 8, 1)
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(x1, cmap='gray')
f(y1, 32, 4, 8)
f(y2, 64, 8, 8)
f(y3, 64, 8, 8)
# plt.imshow(y, cmap='gray')
plt.show()

print(y1.shape)
print(y2.shape)
print(y3.shape)
