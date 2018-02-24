import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


class NeuralNetwork():

    def __init__(self, layer_dim, activation, learning_rate, epochs):
        self. layer_dim = layer_dim
        self.num_layers = len(layer_dim)-1
        self. activation = activation
        self. learning_rate = learning_rate
        self. epochs = epochs
        self.parameters = dict()
        self.cache = dict()
        # Checkers
        self.activation_check()

    def activation_check(self):
        possible = ['relu', 'sigmoid', 'tanh', 'softmax']
        for function in self.activation[1:]:
            if function not in possible:
                print('Activation function not considered in implementation: %s' % function)
                print('Possible activations functions: %s' % ','.join(self.activation))
                exit(1)

    def create_placeholder(self, n_x, n_y, m):
        X = tf.placeholder(tf.float64, shape=[n_x, m])
        Y = tf.placeholder(tf.float64, shape=[n_y, m])
        return X, Y

    def initialize_parameters(self):
        for l in range(1,self.num_layers+1):
            #TODO initialization method, consider different options
            shape_w = (self.layer_dim[l], self.layer_dim[l-1])
            shape_b = (self.layer_dim[l], 1)
            self.parameters['W%s' % str(l)] = tf.get_variable('W%s' % str(l), shape_w, dtype=tf.float64, initializer=tf.random_normal_initializer())
            self.parameters['B%s' % str(l)] = tf.get_variable('B%s' % str(l), shape_b, dtype=tf.float64, initializer=tf.zeros_initializer())

    def forward_propagation(self, A_prev, W, B, activation):
        Z = tf.add(tf.matmul(W, A_prev), B)
        if activation == 'relu':
            A = tf.nn.relu(Z)
        elif activation == 'sigmoid':
            A = tf.nn.sigmoid(Z)
        elif activation == 'softmax':
            A = tf.nn.softmax(Z)
        elif activation == 'tanh':
            A = tf.nn.tanh(Z)
        return A

    def forward_network(self, X):
        W1 = self.parameters['W1']
        B1 = self.parameters['B1']
        W2 = self.parameters['W2']
        B2 = self.parameters['B2']
        A1 = tf.nn.relu(tf.add(tf.matmul(W1, X), B1))
        A = tf.nn.sigmoid(tf.add(tf.matmul(W2, A1), B2))
        return A

    def compute_cost(self, AL, Y):
        predictions = tf.transpose(AL)
        labels = tf.transpose(Y)
        #Cross entropy
        if self.activation[self.num_layers] == 'softmax':
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels)
        elif self.activation[self.num_layers] == 'sigmoid':
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits_v2(logits=predictions, labels=labels)
        #Calculates mean across the cross entropy for each sample.
        cost_function = tf.reduce_mean(cross_entropy)
        return cost_function

    def train(self, input, labels, print_cost=True):
        print('Defining model:')
        # To be able to rerun the model without overwriting tf variables
        ops.reset_default_graph()
        n_x, m = input.shape
        n_y = labels.shape[0]
        costs = list()

        print('\tCreating placeholders for training data: n_x:%s, n_y:%s, m:%s' % (n_x, n_y, m))
        # PLaceholders for data and labels
        X, Y = self.create_placeholder(n_x, n_y, m)

        print('\tInitializing parameters and defining network...')
        self.initialize_parameters()
        AL = self.forward_network(X)
        cost_function = self.compute_cost(AL, Y)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost_function)

        # Initialize variables, gradient descent
        init = tf.global_variables_initializer()

        # Class used to save, restore weight and bias.
        saver = tf.train.Saver()
        save_file = ''

        print('Starting session:')
        with tf.Session() as sess:
            # Initilialize
            sess.run(init)
            for epoch in range(0, self.epochs):
                _, epoch_cost = sess.run([optimizer, cost_function], feed_dict={X: input, Y: labels})
                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    print("\tCost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)


            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()

            # Saving parameters
            self.parameters = sess.run(self.parameters)
            saver.save(sess, save_file)
            # print("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: input, Y: labels}))
