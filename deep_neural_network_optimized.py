import numpy as np
import math
import matplotlib.pyplot as plt
import deep_neural_network_base as dnn
from nn_utils import *


'''
Nerual network with different optimization algorithms.
'''
class NeuralNetwork(dnn.NeuralNetwork):
    def __init__(self, layer_dim, activations, learning_rate, num_iterations, mini_batch_size, optimization, param):
        super(NeuralNetwork, self).__init__(layer_dim, activations, learning_rate, num_iterations, mini_batch_size)
        optimizations = ['momentum', 'adagrad', 'rmsprop', 'adam', 'adadelta', 'adamax']
        if optimization not in optimizations:
            print('Optimizer not contemplated.')
            exit(1)
        if optimization == 'momentum':
            self.optimization = 'momentum'
            self.beta_momentum = param
            self.v = dict()
        elif optimization == 'adagrad':
            self.optimization = 'adagrad'
            self.epsilon_adagrad = param
            self.v = dict()
        elif optimization == 'rmsprop':
            self.optimization = 'rmsprop'
            params = param.split(',')
            self.beta_rmsprop = params[0]
            self.epsilon_rmsprop = params[1]
            self.v = dict()
        elif optimization == 'adam':
            self.optimization = 'adam'
            params = param.split(',')
            self.beta_momentum = params[0]
            self.beta_rmsprop = params[1]
            self.epsilon_adam = params[2]
            self.v = dict()
            self.s = dict()
        elif optimization == 'adadelta':
            self.optimization = 'adadelta'
            params = param.split(',')
            self.beta_adadelta = params[0]
            self.epsilon_adadelta = params[1]
            self.v = dict()
            self.s = dict()
        elif optimization == 'adamax':
            self.optimization = 'adwamax'
            params = param.split(',')
            self.beta_adamax1 = param[0]
            self.beta_adamax2 = param[1]
            self.v = dict()
            self.s = dict()
        # TODO NAG/Nadam

    def initialize_optimizer(self):
        for l in range(1, self.num_layers + 1):
            if self.optimization in ['adam', 'adadelta', 'adamax']:
                self.v['dW%s' % l] = np.zeros(self.parameters['W%s' % l].shape)
                self.s['dW%s' % l] = np.zeros(self.parameters['W%s' % l].shape)
                self.v['dB%s' % l] = np.zeros(self.parameters['B%s' % l].shape)
                self.s['dB%s' % l] = np.zeros(self.parameters['B%s' % l].shape)
            elif self.optimization in ['momentum', 'adagrad', 'rmsprop']:
                self.v['dW%s' % l] = np.zeros(self.parameters['W%s' % l].shape)
                self.v['dB%s' % l] = np.zeros(self.parameters['B%s' % l].shape)

    def update_parameters_momentum(self, t):
        for l in range(1, self.num_layers + 1):
            # MOMENTUM
            # Does a weighted average for a the history points. B^^(x points)~1 --> X points is the approx point for the weighted average.
            # E.g: It will take into account the last 10 gradients.
            self.v['dW%s' % l] = (self.beta_momentum * self.v['dW%s' % l]) + ((1 - self.beta_momentum) * self.grads['dW%s' % l])
            self.v['dB%s' % l] = (self.beta_momentum * self.v['dB%s' % l]) + ((1 - self.beta_momentum) * self.grads['dB%s' % l])
            # Corrected weighted average, this is done to address the imbalance on the first data points since V0=0, the first few points will be off.
            vdw_corrected = self.v['dW%s' % l] / (1 - np.power(self.beta_momentum, t))
            vdb_corrected = self.v['dB%s' % l] / (1 - np.power(self.beta_momentum, t))
            self.parameters['W%s' % l] = self.parameters['W%s' % l] - self.learning_rate * vdw_corrected
            self.parameters['B%s' % l] = self.parameters['B%s' % l] - self.learning_rate * vdb_corrected

    def update_parameters_adagrad(self):
        for l in range(1, self.num_layers + 1):
            self.v['dW%s' % l] += np.power(self.grads['dW%s' % l], 2)
            self.v['dB%s' % l] += np.power(self.grads['dB%s' % l], 2)
            self.parameters['W%s' % l] =  self.parameters['W%s' % l] - self.learning_rate * (self.grads['dW%s' % l]/np.sqrt(self.v['dW%s' % l] + self.epsilon_adagrad))
            self.parameters['B%s' % l] =  self.parameters['B%s' % l] - self.learning_rate * (self.grads['dB%s' % l]/np.sqrt(self.v['dB%s' % l] + self.epsilon_adagrad))

    def update_parameters_adagrad(self):
        for l in range(1, self.num_layers + 1):
            self.v['dW%s' % l] = self.beta_adadelta*self.v['dW%s' % l] + (1-self.beta_adadelta) * np.power(self.grads['dW%s' % l], 2)
            update = - (np.sqrt(self.s['dW%s' % l] + self.epsilon_adadelta)/np.sqrt(self.v['dW%s' % l] + self.epsilon_adadelta)) * self.grads['dW%s' % l]
            self.s['dW%s' % l] = self.beta_adadelta*self.s['dW%s' % l] + (1-self.beta_adadelta) * np.power(update, 2)
            self.parameters['W%s' % l] = self.parameters['W%s' % l] + update

            self.v['dB%s' % l] = self.beta_adadelta * self.v['dB%s' % l] + (1 - self.beta_adadelta) * np.power(self.grads['dB%s' % l], 2)
            update = - (np.sqrt(self.s['dB%s' % l] + self.epsilon_adadelta) / np.sqrt(self.v['dB%s' % l] + self.epsilon_adadelta)) * self.grads['dB%s' % l]
            self.s['dB%s' % l] = self.beta_adadelta * self.s['dB%s' % l] + (1 - self.beta_adadelta) * np.power(update, 2)
            self.parameters['B%s' % l] = self.parameters['B%s' % l] + update

    def update_parameters_rmsprop(self, t):
        for l in range(1, self.num_layers + 1):
            # RMS PROP
            self.v['dW%s' % l] = (self.beta_rmsprop * self.v['dW%s' % l]) + ((1 - self.beta_rmsprop) * np.power(self.grads['dW%s' % l], 2))
            self.v['dB%s' % l] = (self.beta_rmsprop * self.v['dB%s' % l]) + ((1 - self.beta_rmsprop) * np.power(self.grads['dB%s' % l], 2))
            sdw_corrected = self.v['dW%s' % l] / (1 - np.power(self.beta_rmsprop, t))
            sdb_corrected = self.v['dB%s' % l] / (1 - np.power(self.beta_rmsprop, t))
            self.parameters['W%s' % l] = self.parameters['W%s' % l] - self.learning_rate * (np.divide(self.grads['dW%s' % l], np.sqrt(sdw_corrected + self.epsilon_adam)))
            self.parameters['B%s' % l] = self.parameters['B%s' % l] - self.learning_rate * (np.divide(self.grads['dB%s' % l], np.sqrt(sdb_corrected + self.epsilon_adam)))

    def update_parameters_with_adam(self, t, num_epoch):
        for l in range(1, self.num_layers+1):
            # MOMENTUM
            # Does a weighted average for a the history points. B^^(x points)~1 --> X points is the approx point for the weighted average.
            # E.g: It will take into account the last 10 gradients.
            self.v['dW%s' % l] = (self.beta_momentum*self.v['dW%s' % l]) + ((1-self.beta_momentum)*self.grads['dW%s' % l])
            self.v['dB%s' % l] = (self.beta_momentum*self.v['dB%s' % l]) + ((1-self.beta_momentum)*self.grads['dB%s' % l])
            # Corrected weighted average, this is done to address the inbalance on the first data points since V0=0, the first few points will be off.
            vdw_corrected = self.v['dW%s' % l] / (1 - np.power(self.beta_momentum, t))
            vdb_corrected = self.v['dB%s' % l] / (1 - np.power(self.beta_momentum, t))
            # RMS PROP
            self.s['dW%s' % l] = (self.beta_rmsprop*self.s['dW%s' % l]) + ((1-self.beta_rmsprop)*np.power(self.grads['dW%s' % l], 2))
            self.s['dB%s' % l] = (self.beta_rmsprop*self.s['dB%s' % l]) + ((1-self.beta_rmsprop)*np.power(self.grads['dB%s' % l], 2))
            sdw_corrected = self.s['dW%s' % l] / (1 - np.power(self.beta_rmsprop, t))
            sdb_corrected = self.s['dB%s' % l] / (1 - np.power(self.beta_rmsprop, t))

            self.parameters['W%s' % l] = self.parameters['W%s' % l] - self.learning_rate*(np.divide(vdw_corrected, np.sqrt(sdw_corrected + self.epsilon_adam)))
            self.parameters['B%s' % l] = self.parameters['B%s' % l] - self.learning_rate*(np.divide(vdb_corrected, np.sqrt(sdb_corrected + self.epsilon_adam)))

    def update_parameters_adamax(self, t):
        for l in range(1, self.num_layers + 1):
            self.v['dW%s' % l] = self.beta_adamax1 * self.v['dW%s' % l] + (1 - self.beta_adamax1) * np.power(self.grads['dW%s' % l], 2)
            self.v['dB%s' % l] = self.beta_adamax1 * self.v['dB%s' % l] + (1 - self.beta_adamax1) * np.power(self.grads['dB%s' % l], 2)
            self.s['dW%s' % l] = max(self.beta_adamax2*self.s['dW%s' % l], np.abs(self.grads['dW%s' % l]))
            self.s['dB%s' % l] = max(self.beta_adamax2*self.s['dB%s' % l], np.abs(self.grads['dB%s' % l]))
            vdw_corrected = self.v['dW%s' % l] / (1 - np.power(self.beta_adamax1, t))
            vdb_corrected = self.v['dB%s' % l] / (1 - np.power(self.beta_adamax1, t))
            self.parameters['W%s' % l] = self.parameters['W%s' % l] - self.learning_rate*(vdw_corrected/self.s['dW%s' % l])
            self.parameters['B%s' % l] = self.parameters['B%s' % l] - self.learning_rate*(vdb_corrected/self.s['dB%s' % l])

    def update_parameters(self, t, num_epoch):
        if self.optimization == 'momentum':
            self.update_parameters_momentum(t)
        elif self.optimization == 'adam':
            self.update_parameters_with_adam(t, num_epoch)

    def train(self, X_stage, Y_stage, t, num_epoch):
        self.nn_forward(X_stage)
        cost = self.compute_cost(Y_stage)
        self.nn_backwards(Y_stage)
        self.update_parameters(t, num_epoch)
        return cost

    def train_set(self, X, Y, print_cost=True):
        t = 0
        self.initialize_parameters()
        self.initialize_optimizer()
        num_samples = Y.shape[1]
        for i in range(0, self.num_iterations):
            for index in range(0, num_samples, self.mini_batch_size):
                end = index + self.mini_batch_size
                t += 1
                cost = self.train(X[:, index:end], Y[:, index:end], t, i)
                if print_cost and i % 100 == 0:
                    print("Cost after epoch %i:     %f" % (i, cost))
                    self.costs.append(cost)

        if print_cost:
            plt.plot(self.costs)
            plt.ylabel('cost')
            plt.xlabel('iterations (x100)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()
