import numpy as np
from deep_neural_network_batch_normalization import *
from nn_utils import *
from mnist.loader import MNIST


# Loading data.
db = '/Users/aclaudioquiros/Documents/neural_netwoks/Data/MNIST_database/'
mndata = MNIST(db)
images, labels = mndata.load_training()
images_test, labels_test = mndata.load_testing()
images = np.array(images).T
images_test = np.array(images_test).T
labels = onehot(labels, images.shape, sigmoid=True)
labels_test = onehot(labels_test, images_test.shape)
images = normalize(images)
images_test = normalize(images_test)


layer_dim = [images.shape[0], 40, labels.shape[0]]
activations = [None, 'relu', 'sigmoid']

deep_nn = NeuralNetwork(layer_dim, activations, learning_rate=0.001, num_iterations=10, mini_batch_size=1024, eps_norm=1e-7)
deep_nn.gradient_check_nn(images[:, :2], labels[:, :2])



