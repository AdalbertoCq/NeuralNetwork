import numpy as np
import math
import matplotlib.pyplot as plt
from nn_utils import *


class NeuralNetwork():

    def __init__(self, layer_dim, activations, learning_rate, num_iterations, mini_batch_size):
        if len(layer_dim) != len(activations):
            print('Network size doesn\'t match.')
            print('Layer_dim size: %s' % len(layer_dim))
            print('Activation size: %s' % len(activations))
            exit(1)
        self.layer_dim = layer_dim
        self.activations = activations
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.mini_batch_size = mini_batch_size
        self.num_layers = len(self.layer_dim) - 1
        self.parameters = dict()
        self.cache = dict()
        self.grads = dict()

    def initialize_parameters(self):
        # Ranges 1 to num_layers-1
        for l in range(1, self.num_layers + 1):
            # Factor 2 for the ReLU on the Xavier
            if self.activations[l] == 'relu':
                self.parameters['W%s' % l] = np.random.randn(self.layer_dim[l], self.layer_dim[l-1]) / np.sqrt(self.layer_dim[l-1]/2.)
            else:
                self.parameters['W%s' % l] = np.random.randn(self.layer_dim[l], self.layer_dim[l - 1]) / np.sqrt(self.layer_dim[l - 1])
            self.parameters['B%s' % l] = np.zeros((self.layer_dim[l], 1))

    def linear_forward(self, A_prev, W, B, activation):
        Z = np.dot(W, A_prev) + B
        if activation == 'relu':
            A = relu(Z)
        elif activation == 'sigmoid':
            A = sigmoid(Z)
        elif activation == 'softmax':
            A = softmax(Z)
        elif activation == 'tanh':
            A = tanh(Z)
        return A, Z

    def nn_forward(self, X):
        self.cache['A0'] = X
        for l in range(1, self.num_layers+1):
            W = self.parameters['W%s' % l]
            B = self.parameters['B%s' % l]
            activation_layer = self.activations[l]
            self.cache['A%s' % l], self.cache['Z%s' % l] = self.linear_forward(self.cache['A%s' % str(l-1)], W, B, activation_layer)

    def compute_cost(self, Y):
        AL = self.cache['A%s' % str(self.num_layers)]
        # Cross-entropy and Binary cross-entropy cost functions.
        if self.activations[self.num_layers] == 'softmax':
            loss = -np.sum(np.multiply(Y, np.log(AL)), axis=0, keepdims=True)
        elif self.activations[self.num_layers] == 'sigmoid':
            loss = -( np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
        else:
            print('Last activation function not comteplated: %s' % self.activations[-1])
            exit(1)
        cost = np.sum(loss, axis=1, keepdims=True) / float(Y.shape[1])
        cost = np.squeeze(cost)
        return cost

    def linear_backwards(self, dA, A_prev, Z, W, activation):
        if activation == 'relu':
            dZ = np.multiply(dA, relu_derivative(Z))
        elif activation == 'sigmoid':
            dZ = np.multiply(dA, sigmoid_derivative(Z))
        elif activation == 'tanh':
            dZ = np.multiply(dA, tanh_derivative(Z))
        dW = np.dot(dZ, A_prev.T)/float(A_prev.shape[1])
        dB = np.sum(dZ, axis= 1, keepdims=True)/float(A_prev.shape[1])
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, dB

    def nn_backwards(self, Y):
        # Computing L.
        AL = self.cache['A%s' % str(self.num_layers)]
        W = self.parameters['W%s' % str(self.num_layers)]
        A_prev = self.cache['A%s' % str(self.num_layers - 1)]
        dZ = AL - Y
        self.grads['dW%s' % str(self.num_layers)] = np.dot(dZ, A_prev.T)/float(A_prev.shape[1])
        self.grads['dB%s' % str(self.num_layers)] = np.sum(dZ, axis=1, keepdims=True)/float(A_prev.shape[1])
        self.grads['dA%s' % str(self.num_layers - 1)] = np.dot(W.T, dZ)

        # Starts by L-1
        for l in reversed(range(1, self.num_layers)):
            W = self.parameters['W%s' % l]
            Z = self.cache['Z%s' % l]
            dA_prev_temp, dW_temp, db_temp = self.linear_backwards(self.grads['dA%s' % str(l)], self.cache['A%s' % str(l - 1)], Z, W, self.activations[l])
            self.grads['dA%s' % str(l-1)] = dA_prev_temp
            self.grads['dW%s' % l] = dW_temp
            self.grads['dB%s' % l] = db_temp

    def update_parameters(self):
        for l in range(1, self.num_layers+1):
            self.parameters['W%s' % l] = self.parameters['W%s' % l] - self.learning_rate*self.grads['dW%s' % l]
            self.parameters['B%s' % l] = self.parameters['B%s' % l] - self.learning_rate*self.grads['dB%s' % l]

    def train(self, X_stage, Y_stage):
        self.nn_forward(X_stage)
        cost = self.compute_cost(Y_stage)
        self.nn_backwards(Y_stage)
        self.update_parameters()
        return cost

    def train_set(self, X, Y, print_cost=True):
        self.initialize_parameters()
        costs = list()
        num_samples = Y.shape[1]
        for i in range(0, self.num_iterations):
            for index in range(0, num_samples, self.mini_batch_size):
                end = index + self.mini_batch_size
                cost = self.train(X[:, index:end], Y[:, index:end])
                if print_cost and i % 100 == 0:
                    print("Cost after epoch %i:     %f" % (i, cost))
                    costs.append(cost)
        if print_cost:
            plt.figure()
            plt.plot(costs[20:])
            plt.ylabel('Cost Function')
            plt.xlabel('iterations (x100)')
            plt.title('Learning rate =%s' % self.learning_rate)
            plt.show()

    def run_weigths(self, X, Y):
        self.initialize_parameters()
        self.nn_forward(X)
        self.nn_backwards(Y)
        plot_activations(self.cache, self.grads)

    def run(self, X, Y):
        self.nn_forward(X)
        return get_accuracy(self.cache['A%s' % self.num_layers], Y)

    def gradient_check_cost_activation(self, X, Y, epsilon=1e-7):
        print('Loss function')
        x, y = Y.shape
        AL = np.random.standard_normal(Y.shape)
        AL = sigmoid(AL)
        dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

        AL_plus = AL + epsilon
        self.cache['A%s' % str(self.num_layers)] = AL_plus
        cost_plus = self.compute_cost(Y)
        AL_minus = AL - epsilon
        self.cache['A%s' % str(self.num_layers)] = AL_minus
        cost_minus = self.compute_cost(Y)
        grad_approx = (loss_plus-loss_minus)/(2*epsilon)

        numerator = np.linalg.norm(dAL - grad_approx)
        denominator = np.linalg.norm(dAL) + np.linalg.norm(grad_approx)
        difference = np.divide(numerator, denominator)

        if difference > 2e-7:
            print('\tWrong. Difference: %s' % difference)
        else:
            print('\tCorrect. Difference: %s' % difference)
        print('')

        '''
        Checking gradient for each activation function.
        '''
        print('Checking activation functions')
        for activation in ['relu', 'sigmoid', 'softmax']:
            print('Function: %s' % activation)
            Z = X
            if activation == 'relu':
                dA_dZ = relu_derivative(Z)
                A_plus = relu(Z + epsilon)
                A_minus = relu(Z - epsilon)
            elif activation == 'sigmoid':
                dA_dZ = sigmoid_derivative(Z)
                A_plus = sigmoid(Z + epsilon)
                A_minus = sigmoid(Z - epsilon)
            #TODO
            elif activation == 'softmax':
                dA_dZ = softmax_derivative(Z)
                A_plus = softmax(Z + epsilon)
                A_minus = softmax(Z - epsilon)
            grad_approx = (A_plus - A_minus) / (2 * epsilon)
            numerator = np.linalg.norm(dA_dZ - grad_approx)
            denominator = np.linalg.norm(dA_dZ) + np.linalg.norm(grad_approx)  # Step 2'
            difference = np.divide(numerator, denominator)

            if difference > 2e-7:
                print('\tWrong. Difference: %s' % difference)
            else:
                print('\tCorrect. Difference: %s' % difference)
        print('')

    def gradient_check_nn(self, X, Y, epsilon=1e-7):
        '''
        Checking gradient for one step of gradient step.
        '''
        print('Checking gradient for whole nn')
        self.initialize_parameters()
        self.nn_forward(X)
        self.nn_backwards(Y)
        grads = self.grads
        params_dict = self.parameters.copy()
        flatten_parameters_orig, param_sizes = flatten_params(params_dict)
        grad_approx = np.zeros(flatten_parameters_orig.shape)

        index = 0
        size = flatten_parameters_orig.shape[0]
        for index in range(0, size, 1):
            flatten_parameters = flatten_parameters_orig.copy()
            flatten_parameters[index,0] = flatten_parameters_orig[index,0] + epsilon
            reconstructed = reconstruct_params(flatten_parameters, param_sizes)
            self.parameters = reconstructed.copy()
            self.nn_forward(X)
            cost_plus = self.compute_cost(Y)

            flatten_parameters = flatten_parameters_orig.copy()
            flatten_parameters[index, 0] = flatten_parameters_orig[index, 0] - epsilon
            reconstructed = reconstruct_params(flatten_parameters, param_sizes)
            self.parameters = reconstructed.copy()
            self.nn_forward(X)
            cost_minus = self.compute_cost(Y)

            grad_approx[index,0] = (cost_plus-cost_minus)/(2*epsilon)
            flatten_parameters[index, 0] += epsilon

        grads_reconstructed = reconstruct_params(grad_approx, param_sizes)
        for param_o in grads_reconstructed:
            param = 'd%s' % param_o
            print('Checking parameter: %s' % param)
            grad_param = grads[param]
            grad_param_approx = grads_reconstructed[param_o]
            numerator = np.linalg.norm(grad_param - grad_param_approx)
            denominator = np.linalg.norm(grad_param) + np.linalg.norm(grad_param_approx)
            difference = np.divide(numerator, denominator)
            if difference > 2e-7:
                print('\tWrong. Difference: %s' % difference)
            else:
                print('\tCorrect. Difference: %s' % difference)

