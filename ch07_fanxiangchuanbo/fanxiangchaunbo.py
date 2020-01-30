import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
import sklearn.datasets as skd

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

x, y = skd.make_moons(n_samples=2000, noise=0.2, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x.shape, y.shape)
print(x_train.shape)

plt.figure(figsize=(16, 12))
plt.title("Classification Dataset Visualization ", fontsize=30)
axes = plt.gca()
axes.set(xlabel="x1", ylabel="x2")
plt.scatter(x[:, 0], x[:, 1],c=y, s=40, cmap=plt.cm.Spectral,edgecolors='none')
plt.savefig('moonsdataset.svg')
plt.show()

class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r,0)
        elif self.activation == 'tahn':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r

class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def backpropagation(self, X, y, learning_rate):
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        for i in range(len(self._layers)):
            layer = self._layers[i]
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * o_i.T * learning_rate

    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []
        for i in range(max_epochs):
            for j in range(len(X_train)):
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
                print('Accuracy: %.2f%%' % (self.accuracy(self.predict(X_test), y_test.flatten()) * 100))
        return mses

nn = NeuralNetwork()
nn.add_layer(Layer(2, 25, 'sigmoid'))
nn.add_layer(Layer(25, 50, 'sigmoid'))
nn.add_layer(Layer(50, 25, 'sigmoid'))
nn.add_layer(Layer(25, 2, 'sigmoid'))

m = nn.train(x_train, x_test, y_train, y_test, 0.02, 1000)

plt.figure()
n = [i * 10 for i in range(len(m))]
plt.plot(n, m)
plt.ylabel('MSE')
plt.xlabel('Step')
plt.show()