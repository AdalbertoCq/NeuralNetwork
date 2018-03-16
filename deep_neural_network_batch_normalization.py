import numpy as np
import math
import matplotlib.pyplot as plt
import deep_neural_network_base as dnn
from nn_utils import *


class NeuralNetwork(dnn.NeuralNetwork):
    def __init__(self, layer_dim, activations, learning_rate, num_iterations, mini_batch_size , eps_norm):
        super(NeuralNetwork, self).__init__(layer_dim, activations, learning_rate, num_iterations, mini_batch_size)
        self.eps_norm = eps_norm
        self.test = True
        self.mean = None
        self.variance = None

    def initialize_parameters(self):
        # Ranges 1 to num_layers-1
        for l in range(1, self.num_layers + 1):
            self.parameters['W%s' % l] = np.random.randn(self.layer_dim[l], self.layer_dim[l-1]) * np.sqrt(2.0/self.layer_dim[l-1])
            self.parameters['G%s' % l] = np.ones((self.layer_dim[l], 1))
            self.parameters['B%s' % l] = np.zeros((self.layer_dim[l], 1))

    def normalize_forward(self, Z, G, B):
        Z_norm = normalize(Z, self.eps_norm, self.test, )
        Y = (G*Z_norm) + B
        return Y

    def linear_forward(self, A_prev, G, B, W, activation):
        Z = np.dot(W, A_prev)
        Y = self.normalize_forward(Z, G, B)
        if activation == 'relu':
            A = relu(Y)
        elif activation == 'sigmoid':
            A = sigmoid(Y)
        elif activation == 'softmax':
            A = softmax(Y)
        return A, Y, Z

    def nn_forward(self, X):
        self.cache['A0'] = X
        for l in range(1, self.num_layers+1):
            W = self.parameters['W%s' % l]
            G = self.parameters['G%s' % l]
            B = self.parameters['B%s' % l]
            activation_layer = self.activations[l]
            self.cache['A%s' % l], self.cache['Y%s' % l], self.cache['Z%s' % l] = self.linear_forward(self.cache['A%s' % str(l-1)], G, B, W, activation_layer)

    def normalize_backwards(self, dY, Z, G):
        m = float(Z.shape[1])
        mean = np.mean(Z, axis=1, keepdims=True)
        var = np.var(Z, axis=1, keepdims=True)
        std = 1./np.sqrt(var + self.eps_norm)

        dZ_norm = dY * G
        Z_norm = normalize(Z, self.eps_norm)
        first_term = m * dZ_norm
        second_term = np.sum(dZ_norm, axis=1, keepdims=True)
        third_term_a = (Z - mean) * (np.power(std, 2))
        third_term_b = np.sum(dZ_norm * (Z - mean), axis=1, keepdims=True)
        third_term = third_term_a * third_term_b
        dZ = (1./m) * std * (first_term - second_term - third_term)
        dG = np.sum(dY * Z_norm, axis=1, keepdims=True)/m
        dB = np.sum(dY, axis=1, keepdims=True)/m

        return dZ, dG, dB

    def linear_backwards(self, dA, A_prev, Y, Z, G, W, activation):
        m = float(Z.shape[1])
        if activation == 'relu':
            dY = np.multiply(dA, relu_derivative(Y))
        elif activation == 'sigmoid':
            dY = np.multiply(dA, sigmoid_derivative(Y))
        dZ, dG, dB = self.normalize_backwards(dY, Z, G)
        dW = np.dot(dZ, A_prev.T)/m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dG, dB, dW

    def nn_backwards(self, Y):
        # Computing L.
        AL = self.cache['A%s' % str(self.num_layers)]
        Z = self.cache['Z%s' % str(self.num_layers)]
        W = self.parameters['W%s' % str(self.num_layers)]
        G = self.parameters['G%s' % str(self.num_layers)]
        activation_layer = self.activations[self.num_layers]
        A_prev = self.cache['A%s' % str(self.num_layers - 1)]

        if self.activations[-1] == 'softmax':
            dY = AL - Y
        elif self.activations[-1] == 'sigmoid':
            dAL = - (1. / AL.shape[1]) * np.divide(Y, AL)
            dY = np.multiply(dAL, sigmoid_derivative(Y))
        elif self.activations[-1] == 'relu':
            dAL = - (1. / AL.shape[1]) * np.divide(Y, AL)
            dY = np.multiply(dAL, relu_derivative(Y))
        elif self.activations[-1] == 'tanh':
            dAL = - (1. / AL.shape[1]) * np.divide(Y, AL)
            dY = np.multiply(dAL, tanh_derivative(Y))

        dZ, dG_temp, db_temp = self.normalize_backwards(dY, Z, G)
        dW_temp = np.dot(dZ, A_prev.T)/float(A_prev.shape[1])
        dA_prev_temp = np.dot(W.T, dZ)
        self.grads['dA%s' % str(self.num_layers - 1)] = dA_prev_temp
        self.grads['dG%s' % str(self.num_layers)] = dG_temp
        self.grads['dB%s' % str(self.num_layers)] = db_temp
        self.grads['dW%s' % str(self.num_layers)] = dW_temp

        # Starts by L-1
        for l in reversed(range(1, self.num_layers)):
            W = self.parameters['W%s' % l]
            G = self.parameters['G%s' % l]
            Z = self.cache['Z%s' % l]
            Y = self.cache['Y%s' % l]
            activation_layer = self.activations[l]
            dA_prev_temp, dG_temp, db_temp, dW_temp = self.linear_backwards(self.grads['dA%s' % str(l)], self.cache['A%s' % str(l - 1)], Y, Z, G, W, activation_layer)
            self.grads['dA%s' % str(l-1)] = dA_prev_temp
            self.grads['dG%s' % l] = dG_temp
            self.grads['dB%s' % l] = db_temp
            self.grads['dW%s' % l] = dW_temp

    def update_parameters(self):
        for l in range(1, self.num_layers+1):
            self.parameters['W%s' % l] = self.parameters['W%s' % l] - self.learning_rate*self.grads['dW%s' % l]
            self.parameters['G%s' % l] = self.parameters['G%s' % l] - self.learning_rate*self.grads['dG%s' % l]
            self.parameters['B%s' % l] = self.parameters['B%s' % l] - self.learning_rate*self.grads['dB%s' % l]


    def run(self, X, Y, mean, variance):
        self.test = True
        self.mean = mean
        self.variance = variance
        self.nn_forward(X)
        accuracy = get_accuracy(self.cache['A%s' % self.num_layers], Y)
        self.test = False
        self.mean = None
        self.variance = None
        return accuracy