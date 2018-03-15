import h5py
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import keras as K


def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')


def define_model(dropout_prob, opt, weigth_init, lr):
    model = Sequential()
    # Flattening the images 28x28 to vectors 784 elements.
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(512, kernel_initializer=weigth_init, activation='relu'))
    model.add(Dropout(dropout_prob))
    model.add(Dense(512, kernel_initializer=weigth_init, activation='relu'))
    model.add(Dropout(dropout_prob))
    model.add(Dense(10, kernel_initializer=weigth_init, activation='softmax'))
    # Model summary
    # model.summary()
    # compile the model
    opt = opt.lower()
    if opt == 'sgd':
        opt_model = K.optimizers.SGD(lr=lr)
    elif opt == 'rmsprop':
        opt_model = K.optimizers.RMSprop(lr=lr)
    elif opt == 'adagrad':
        opt_model = K.optimizers.Adagrad(lr=lr)
    elif opt == 'adadelta':
        opt_model = K.optimizers.Adadelta(lr=lr)
    elif opt == 'adam':
        opt_model = K.optimizers.Adam(lr=lr)
    elif opt == 'adamax':
        opt_model = K.optimizers.Adamax(lr=lr)
    elif opt == 'nadam':
        opt_model = K.optimizers.Nadam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt_model, metrics=['accuracy'])
    return model


# use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))

# Rescale images to (0,1) range.
# Not constraining the data between (0, 1) went from train=99%, test=97% to train=19.12%, test=19.43%. Huge change.
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Convert labels to one hot.
y_test = np_utils.to_categorical(y_test, num_classes=10)
y_train = np_utils.to_categorical(y_train, num_classes=10)

# keep track of the best model.
checkpointer = ModelCheckpoint(filepath='/Users/aclaudioquiros/Documents/PycCharm/Neural Network Python/mnist.model.best.hdf5',  verbose=1, save_best_only=True)

# '''
# Intent of this piece of code is to see the impact of the different optimizations, weight initialization and batch sizes for the same NN architecture.
# '''
# loss_histories = dict()
# # Impact of different optimizers, how far can they reach for the same number of epochs and learning rate.
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
for opt in optimizers:
    # Model definition.
    model = define_model(0.2, opt, 'he_normal', 0.002)
    hist = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_split=0.2, verbose=0, shuffle=True)
    loss_histories[opt] = hist
    plot = plt.plot(hist.epoch, hist.history['loss'], label=opt)
    # evaluate test accuracy
    score_train = model.eva2luate(X_train, y_train, verbose=0)
    score_test = model.evaluate(X_test, y_test, verbose=0)
    accuracy_train = 100*score_train[1]
    accuracy_test = 100*score_test[1]
    # print test accuracy
    print('Optimizer: %s Train accuracy: %.4f%% Test accuracy: %.4f%%' % (opt, accuracy_train, accuracy_test))
# Need to plot the loss trend, this will show how fast the different optimizations impact the learning curve.
plt.title('Loss function: categorical_crossentropy')
plt.xlabel('Epochs')
plt.legend(optimizers)
plt.show()


# batch sizes and epochs
batch_size = [16, 32, 64, 128, 256, 512]
epochs = [10, 50, 100]
for epoch in epochs:
    for bs in batch_size:
        # Model definition.
        model = define_model(0.2, 'adam', 'he_normal', 0.002)
        hist = model.fit(X_train, y_train, batch_size=bs, epochs=epoch, validation_split=0.2, verbose=0, shuffle=True)

        # evaluate test accuracy
        score_train = model.evaluate(X_train, y_train, verbose=0)
        score_test = model.evaluate(X_test, y_test, verbose=0)
        accuracy_train = 100 * score_train[1]
        accuracy_test = 100 * score_test[1]
        # print test accuracy
        print('Epochs: %s Batch_size: %s Train accuracy: %.4f%% Test accuracy: %.4f%%' % (epoch, bs, accuracy_train, accuracy_test))

# Weight initialization.
loss_histories_w = dict()
init_mode = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
for weigth_init in init_mode:
    # Model definition.
    model = define_model(0.2, 'adam', weigth_init, 0.002)
    hist = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_split=0.2, verbose=0, shuffle=True)
    loss_histories_w[weigth_init] = hist
    plot = plt.plot(hist.epoch, hist.history['loss'], label=weigth_init)
    # evaluate test accuracy
    score_train = model.evaluate(X_train, y_train, verbose=0)
    score_test = model.evaluate(X_test, y_test, verbose=0)
    accuracy_train = 100*score_train[1]
    accuracy_test = 100*score_test[1]
    # print test accuracy
    print('Initilization: %s Train accuracy: %.4f%% Test accuracy: %.4f%%' % (weigth_init, accuracy_train, accuracy_test))
# Need to plot the loss trend, this will show how fast the different optimizations impact the learning curve.
plt.title('Loss function: categorical_crossentropy')
plt.xlabel('Epochs')
plt.legend(init_mode)
plt.show()


'''
Intent of this piece of code is to do a coarse search of the learning rate and regularization for a decided NN architecture.
'''

# Once seen and decided which optimization, weight initialization and bacth_size.
# Search for optimal learning_rate and dropout/L reg.
# Doing this randomly instead of a grid search, that maximaxes the options of reaching the minima.
# Also, search for learning rate and reg is done in log scale since the impact is not linear.
max_count = 100
for count in range(0, max_count):
    dropout = 10**np.random.uniform(-1, 0)
    lr = 10**np.random.uniform(-3, 0)
    model = define_model(dropout, 'adam', 'he_uniform', lr)
    hist = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2, verbose=0, shuffle=True)
    # evaluate test accuracy
    score_train = model.evaluate(X_train, y_train, verbose=0)
    score_test = model.evaluate(X_test, y_test, verbose=0)
    accuracy_train = 100 * score_train[1]
    accuracy_test = 100 * score_test[1]
    # print test accuracy
    print('Lr: %s Dropout: %s Train accuracy: %.4f%% Test accuracy: %.4f%%' % (lr, dropout, accuracy_train, accuracy_test))