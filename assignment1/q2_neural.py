#!/usr/bin/env python

import numpy as np
import random

from cs224.assignment1.q1_softmax import softmax
from cs224.assignment1.q2_sigmoid import sigmoid, sigmoid_grad
from cs224.assignment1.q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # (M x Dx) * (Dx x H) = (M x H)
    hidden_layer = sigmoid(data.dot(W1) + b1)
    # (M x H) * (H x Dy) = (M x Dy)
    output = softmax(hidden_layer.dot(W2) + b2)
    cost = -1 * (labels * np.log(output)).sum(axis=1)
    cost = cost.sum()

    def check_shape(x, grad):
        assert x.shape == grad.shape, '{}, {}'.format(x.shape, grad.shape)

    gradsfm = output - labels  # (M x Dy). We know d(CE)/d(theta) = yhat - y
    gradb2 = gradsfm.sum(axis=0, keepdims=True)  # (1 x Dy)
    check_shape(b2, gradb2)
    gradW2 = hidden_layer.T.dot(gradsfm)  # (H x M) * (M x Dy) = (H x Dy)
    check_shape(W2, gradW2)

    gradh = gradsfm.dot(W2.T)  # (M x Dy) * (Dy x H) = (M x H)
    gradsig = gradh * hidden_layer * (1 - hidden_layer)
    gradb1 = gradsig.sum(axis=0, keepdims=True)  # (1 x H)
    check_shape(b1, gradb1)

    gradW1 = data.T.dot(gradsig)  # (Dx x M) * (M x H) = (Dx x H)
    check_shape(W1, gradW1)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def nn_sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]  # params length = 50 + 5 + 50 + 10 = 115
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_nn_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    nn_sanity_check()
    your_nn_sanity_checks()
