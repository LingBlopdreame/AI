import tensorflow as tf
import matplotlib.pyplot as plt

class Liner(object):
    def __init__(self):
        self.W = tf.Variable(10.0)
        self.b = tf.Variable(5.0)

    def __call__(self, x):
        return x * self.W + self.b

def loss(predict_y, target_y):
    return tf.reduce_mean(tf.square(predict_y - target_y))

liner = Liner()

W = 3.0
b = 2.0
inputs  = tf.random.normal(shape=[5000])
noise   = tf.random.normal(shape=[5000])
outputs = inputs * W + b + noise

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, liner(inputs), c='r')
plt.show()
print('Current loss: %1.6f' % loss(liner(inputs), outputs).numpy())

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(liner(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


losses = []
Ws, bs = [], []
for epoch in range(100):
    Ws.append(liner.W.numpy())
    bs.append(liner.b.numpy())
    current_loss = loss(liner(inputs), outputs)
    train(liner, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch, Ws[-1], bs[-1], current_loss))

plt.figure(1)
plt.plot(range(100), Ws, 'r', range(100), bs, 'b')
plt.plot([W] * 100, 'r--', [b] * 100, 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])

plt.figure(2)
plt.plot(inputs, outputs, 'ro')
plt.plot(inputs, liner(inputs), 'b')

plt.show()
