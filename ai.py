"""
The goal is to be able to create any size - any purpose neural networks out of having only the list of layer sizes (e.g [8, 4, 2]) and layer types (activation function for each layer).

Usage example:
NeuralNetwork([784, 100, 15, 10], [lambda x: x, ai.Sigmoid, ai.BinaryStep, ai.Softmax])
# Note: Custom activation functions can be used as well, as long as they are callable and accept a numpy array as input.

Offered Classes:
Layer(size, activation_function))
WeightMatrix(layer1, layer2)
NeuralNetwork(layer_sizes, layer_types)

Offered Functions:
BinaryStep(values)
Sigmoid(values)
Softmax(values)
ReLU(values)
"""

import numpy as np
from typing import Callable


class Layer:
    """
    One layer in a neural network
    """
    def __init__(self, size : int, activation_function : Callable):
        # TODO
        pass

class WeightMatrix:
    """
    Handles weights between 2 layers
    """
    def __init__(self, layer1 : Layer, layer2 : Layer):
        # TODO
        pass

class NeuralNetwork:
    """
    Handles multiple layers and interconnects them using multiple WeightMatrix objects
    """
    # TODO: learning rate annealing - Decrease the learning rate as the training progresses
    def __init__(self, layer_sizes : list[int], layer_types : list[Callable]):
        # TODO
        pass


def BinaryStep(values):
    """
    Binary step activation function
    """
    # TODO
    pass


def Sigmoid(values):
    """
    Sigmoid activation function
    """
    # TODO
    pass


def Softmax(values):
    """
    Softmax activation function
    """
    # TODO
    pass


def ReLU(values):
    """
    ReLU activation function
    """
    # TODO
    pass