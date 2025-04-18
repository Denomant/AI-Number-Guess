"""
The goal is to be able to create any size - any purpose neural networks out of having only the list of layer sizes (e.g [8, 4, 2]) and layer types (activation function for each layer).

Usage example:
NeuralNetwork([784, 100, 20, 15, 10], [lambda x: x, ai.Sigmoid, ai.ReLU ,ai.BinaryStep, ai.Softmax])
# Note: Custom activation functions can be used as well, as long as they are callable, accept a numpy array as input and return a numpy array as output.

Offered Classes:
Layer(size, activation_function)
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


ActivationFunction = Callable[[np.ndarray], np.ndarray] # Function that can be considered as an activation function

class Layer:
    """
    One layer in a neural network
    """
    def __init__(self, size : int, activation_function : ActivationFunction):
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
    def __init__(self, layer_sizes : list[int], layer_types : list[ActivationFunction]):
        assert len(layer_sizes) == len(layer_types), "Layer sizes and types must be the same length"


def BinaryStep(values : np.ndarray) -> np.ndarray:
    """
    Binary step activation function
    if number is positive, return 1, else return 0
    """
    return np.where(values > 0, 1, 0)


def Sigmoid(values : np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function
    returns a number between 0 and 1 for any number
    """
    bounded = np.clip(values, -500, 500)
    return 1 / (1 + np.exp(-bounded))


def Softmax(values : np.ndarray) -> np.ndarray:
    """
    Softmax activation function
    Converts a vector of numbers into a probability distribution
    """
    exp_x = np.exp(values - np.max(values))
    return exp_x / np.sum(exp_x, axis=0)


def ReLU(values : np.ndarray) -> np.ndarray:
    """
    ReLU activation function
    if number is positive, return number, else return 0
    """
    return np.maximum(0, values)