#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

import sys
import nnet

def run():
    # Fetch data
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='../data')

    split = 60000
    X_train = np.reshape(mnist.data[:split], (-1, 1, 28, 28))/255.0
    y_train = mnist.target[:split]
    n_classes = np.unique(y_train).size

    # Downsample training data
    n_train_samples = 5000
    train_idxs = np.random.random_integers(0, split-1, n_train_samples)
    X_train = X_train[train_idxs, ...]
    y_train = y_train[train_idxs, ...]

    # Downsample training data
    n_test_samples = 1000
    test_idxs = np.random.random_integers(split, mnist.data.shape[0]-1, n_test_samples)
    X_test = np.reshape(mnist.data[test_idxs, ...], (-1, 1, 28, 28))/255.
    y_test = mnist.target[test_idxs]

    # Setup convolutional neural network
    nn = nnet.NeuralNetwork(
        layers=[
            nnet.Conv(
                n_feats=12,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            nnet.Activation('relu'),
            nnet.Pool(
                pool_shape=(2, 2),
                strides=(2, 2),
                mode='max',
            ),
            nnet.Conv(
                n_feats=16,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            nnet.Activation('relu'),
            nnet.Flatten(),
            nnet.Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.02,
            ),
            nnet.LogRegression(),
        ],
    )

    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, X_test, y_test, learning_rate=0.05, max_iter=3, batch_size=32, test_iter=3)
    t1 = time.time()
    print('Training Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)

    ## visualize the filters
    #W,b = nn.layers[0].params()
    #vis_square(W.transpose(1,2,3,0))

    nn.call_plot(flag = True)
    
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if data.shape[2]==1:
        data = np.tile(data,3)
    plt.imshow(data)
    plt.draw()
    plt.show()
    
if __name__ == '__main__':
    run()
