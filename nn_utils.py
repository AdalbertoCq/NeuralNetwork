import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib

def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    a = z * (z>0)
    return a


def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a


def softmax(z):
    log_c = np.max(z, axis=0, keepdims=True)
    z_exp = np.exp(z-log_c)
    z_sum = np.sum(z_exp, axis=0, keepdims=True)
    a = np.divide(z_exp, z_sum)
    return a


def softmax_derivative(z):
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp, axis=0, keepdims=True)
    a = np.divide(z_exp, z_sum)
    da = a * (1-z_sum)
    return da


def relu_derivative(z):
    ds = 1 * (z > 0)
    return ds


def sigmoid_derivative(z):
    da = np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    return da


def normalize(Z, eps_norm=1e-8):
    mean = np.mean(Z, axis=1, keepdims=True)
    variance = np.var(Z, axis=1, keepdims=True)
    Z_norm = np.divide((Z - mean), np.sqrt(variance+eps_norm))
    return Z_norm


def mean_derivative(Z):
    dmean = np.ones(Z.shape)*(1./Z.shape[1])
    return dmean


def variance_derivative(Z):
    mean = np.mean(Z, axis=1, keepdims=True)
    dvar_dz = np.ones(Z.shape)*(2. * (Z-mean))/Z.shape[1]
    return dvar_dz


def flatten_params(params):
    param_sizes = OrderedDict()
    flatten_parameters = list()
    for parameter in params:
        param_sizes[parameter] = dict()
        param_sizes[parameter]['shape'] = dict()
        param_sizes[parameter]['shape']['x'] = params[parameter].shape[0]
        param_sizes[parameter]['shape']['y'] = params[parameter].shape[1]
        flatten_parameters.extend(params[parameter].ravel())
    flatten_parameters = np.array(flatten_parameters).reshape(len(flatten_parameters), 1)
    return flatten_parameters, param_sizes


def reconstruct_params(flatten_parameters, param_sizes):
    reconstructed = dict()
    index = 0
    for param in param_sizes:
        x = param_sizes[param]['shape']['x']
        y = param_sizes[param]['shape']['y']
        range = x * y
        something = flatten_parameters[index: index + range, 0].reshape(x,y)
        reconstructed[param] = something
        index += range
    return reconstructed


def onehot(labels, shape, sigmoid=False):
    output_labels = np.zeros((max(labels)+1, shape[1]))
    if sigmoid:
        output_labels = np.zeros((1, shape[1]))
    index=0
    for item in labels:
        if sigmoid:
            if int(item) == 1:
                output_labels[0][index] = 1
        else:
            output_labels[int(item)][index] = 1
        index+=1
    return output_labels


def get_accuracy(predictions, labels):
    max_predictions = np.max(predictions, axis=0, keepdims=True)
    predictions = ((predictions /max_predictions) == 1) * 1
    matches = np.sum(predictions * labels)
    return (float(matches)/float(labels.shape[1])) * 100


def plot_activations(cache, grads):
    matplotlib.rcParams.update({'font.size': 8})
    activations = [el for el in cache.keys() if 'A' in el]
    f, array = plt.subplots(4, len(activations))
    act_means = list()
    act_stds = list()
    for ind in range(0, len(activations)):
        mean = np.mean(cache['A%s' % str(ind)])
        std = np.std(cache['A%s' % str(ind)])
        print('A%s mean/std: %s %15s' % (str(ind), mean, std))
        act_means.append(mean)
        act_stds.append(std)

    array[0, 0].plot(range(len(activations)), act_means)
    array[0, 0].set_xlabel('Layer')
    array[0, 0].set_title('Mean')
    array[0, 1].plot(range(len(activations)), act_stds)
    array[0, 1].set_xlabel('Layer')
    array[0, 1].set_title('Std')

    for ind in range(0, len(activations)):
        array[1, ind].hist(cache['A%s' % str(ind)].ravel(), 30, range=(-3,3))
        array[1, ind].set_title('Layer %s' % ind)

    dactivations = [el for el in grads.keys() if 'dA' in el]
    dact_means = list()
    dact_stds = list()
    for ind in range(0, len(dactivations)):
        dmean = np.mean(grads['dA%s' % str(ind)])
        dstd = np.std(grads['dA%s' % str(ind)])
        print('dA%s mean/std: %s %15s' % (str(ind), dmean, dstd))
        dact_means.append(dmean)
        dact_stds.append(dstd)

    array[2, 0].plot(range(len(dactivations)), dact_means)
    array[2, 0].set_xlabel('Layer')
    array[2, 0].set_title('Mean')
    array[2, 1].plot(range(len(dactivations)), dact_stds)
    array[2, 1].set_xlabel('Layer')
    array[2, 1].set_title('Std')

    for ind in range(0, len(dactivations)):
        array[3, ind].hist(grads['dA%s' % str(ind)].ravel(), 30, range=(-3, 3))
        array[3, ind].set_title('Layer %s' % ind)

    f.subplots_adjust(hspace=0.5)
    plt.show()

