import tensorflow as tf
import numpy as np
import datetime

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_onehot, y_test_onehot = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)
x_train = tf.expand_dims(x_train, axis=3)

network = tf.keras.Sequential([tf.keras.layers.Conv2D(6, [3,3], strides=1),
                               tf.keras.layers.MaxPooling2D([2,2], strides=2),
                               tf.keras.layers.ReLU(),
                               tf.keras.layers.Conv2D(16, [3,3], strides=1),
                               tf.keras.layers.MaxPooling2D([2,2], strides=2),
                               tf.keras.layers.ReLU(),
                               tf.keras.layers.Flatten(),
                               tf.keras.layers.Dense(120, activation='relu'),
                               tf.keras.layers.Dense(84, activation='relu'),
                               tf.keras.layers.Dense(10),
                               ])
network.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=['accuracy']
                )

log_dir="G:\\AI\\ch10_CNN\\logs\\first\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# network.build(input_shape=[4,28,28,1])
# network.summary()

network.fit(x=x_train,
            y=y_train_onehot,
            batch_size=512,
            verbose=1,
            epochs=50,
            callbacks=[tensorboard_callback]
            )

# criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#
# with tf.GradientTape() as tape:
#     x = tf.expand_dims(x, axis=3)
#     y_onehot = tf.one_hot(y, depth=10)
#     loss = criteon(y_onehot, out)
# grads = tape.gradient(loss, network.trainable_variables)
# tf.keras.optimizers.apply_gradient(zip(grads, network.trainable_variables))



# x = tf.random.normal([2,5,5,3])
# w = tf.random.normal([3,3,3,4])
#
# out = tf.nn.conv2d(x, w, strides=1, padding=[[0,0],[0,0],[0,0],[0,0]])
# print(out)


