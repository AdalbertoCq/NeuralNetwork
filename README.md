# NeuralNetwork
Neural Network implementation in Numpy & Keras. 

The code consists in the following (The idea is that in the future this can be combined, as of now the are used independently): 
* deep_neural_network_base: Class that will build a MLP based on the input activation functions per layer and number of nodes.
* deep_neural_network_batch_normalization: Extension of deep_neural_network_base class, that implements batch normalization.
* deep_neural_network_dropout: Extension of deep_neural_network_base class that implements dropout for training and inference.
* deep_neural_network_l2regularization: Extension of deep_neural_network_base class that implements L2 regularization.
* deep_neural_network_optimized: Extension of deep_neural_network_base class that given the following options for gradient update:
    * Momentum.
    * Nag: pending.
    * Adagrad.
    * RMSprop.
    * Adadelta.
    * Adam.
    * Adamax.
    * Nadam: pending.
    
* launch_nn_training: Code to test out the implementation on MNIST data.
* deep_neural_network_keras: Keras implementation of an MLP for MNIST, used to do a comprehensive analysis on the explanations below.

This is a brief summary of my own understanding for: Regularization, Optimizations, Batch Normalization and Gradient updates.


## _**Comparison between batch/SGD/mini-batch:**_

* Mathematical demostration: [Backpropagation](https://github.com/AdalbertoCq/NeuralNetwork/blob/master/Plots%20%26%20Docs/backprop.PDF)
* Mathematical demostration: [Cross-entropy & Softmax gradients](https://github.com/AdalbertoCq/NeuralNetwork/blob/master/Plots%20%26%20Docs/cross_entropy_softmax.PDF)

### Batch:
* It doesn’t allow on-line training.
* Slow update.

### SGD:
* Allows on-line training.
* Updates only on one sample which cause to have high variance in the cost function with the weight updates.
* Complicates getting to the minima as it overshoots constantly.

### Mini-batch:
* With a proper size reduces the variances on the updates.
* Allows a faster converge to a minima since updates sooner than batch.
* A good matrix size cam take advantage of different computer architectures for matrix operations.

## _**Batch Normalization:**_
* Limits the amount that the data distribution in the layer can shift affected by weight and bias updates.
* It weakens the coupling between one layer and the previous one. 
* It has the same effect as normalizing the inputs, it will control the data distribution from getting really high or really low values through the network. It allows a faster convergence to the minima.
* Includes a regularization effect since in training the normalization is done through the mini-batch, including some noise that translates in the network activation outputs Z[l]. Similar to Dropout in effect.
* Mathematical demostration: [Batch Normalization Gradient](https://github.com/AdalbertoCq/NeuralNetwork/blob/master/Plots%20%26%20Docs/batch_norm_backprop.PDF)


## _**Regularization:**_
* Techniques to prevent the neural network to overfit on the training data.
### L2:
* Forces the weights to take lower values and preventing high standouts for the weight values.
* Penalizes the Cost function with the median quadratic values for the weights.
* Prevents weights to take large values, this helps to generalize features, not giving to much importance to some compare to others:
  * When the weight is updated, there's a certain weight decay introduced from the gradient of Cost function (Added median quadratic values in cost function).

### L1:
* It penalizes the weight with the absolute value.
* Same intention as L2, but lower impact.

### Dropout:
* Randomly drops activation outputs along the network, only during training and based on a certain probability.
* The objective is to balance the weight values preventing them to fit only on the training set samples and not for the general case.


## _**Convergence to minima on cost function optimizations:**_
1. Momentum:
    * Uses previous gradients (cost function slopes) to overcome possible local minima valley.
    * Computes a weighted average of previous and current gradient.

2. Nesterov accelerated gradient:
    * TODO

3. Adagrad:
    * Normalizes the weight updates; dividing by the the variance of the weight updates.
    * Gradients with larger updates will effectively receive a smaller update.
    * The monotonic learning rate proves to be too aggressive and stops learning sooner.

4. RMSprop:
    * Adjust the aggressiveness of the monotonically decreasing learning rate in Adagrad.
    * Unlike in Adagrad, the updates do not get monotonically smaller.

5. Adadelta:
    * Based on Adagrad. Improves its problem with a converging to 0 update over time.
    * Based on two main ideas:
      * Scale learning rate based on historical gradient that only takes into account a given window of updates.
      * Use component that serves as an acceleration term, as in momentum.

6. Adam:
    * Combines Momentum and RMS prop.
    * Commonly used a default optimization algorithm.
    * It also includes the bias correction for the initial values, that way at the beginning it isn't so off.

7. Adamax:
    * Similar to Adam.
    * Changes over the learning rate factor. When the update is small, it is ignored --> This makes it more robust to noisy gradients.

8. Nadam:
    * TODO

## _**Loss function over different optimizations:**_
<img src="https://github.com/AdalbertoCq/NeuralNetwork/tree/master/Plots_n_Docs/Optimization_plots.png"/>

* Learning rate = 0.002
* Epochs = 25
* Batch size = 128
* Dropout prob = 0.2
* Weight Initialization = He over normal distribution.


