import sys
sys.path.append('/Users/aclaudioquiros/Documents/PycCharm/Neural Network Tensorflow/')
from deep_neural_network import *
from mnist import MNIST
import numpy as np
from nn_utils import *


mndata = MNIST('/Users/aclaudioquiros/Documents/neural_netwoks/Data/MNIST_database/')
images, labels = mndata.load_training()
images_test, labels_test = mndata.load_testing()
images = np.array(images).T
images_test = np.array(images_test).T
# images = normalize(images.T, 1e-8)
# images_test = normalize(images_test, 1e-8)
labels = onehot(labels, images.shape)
labels_test = onehot(labels_test, images_test.shape)

layer_dim = [images.shape[0], 20, labels.shape[0]]
activation = [None, 'relu', 'softmax']
dnn = NeuralNetwork(layer_dim, activation, learning_rate=0.5, epochs=4000)
dnn.train(images, labels)