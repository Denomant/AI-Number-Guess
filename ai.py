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
from typing import Callable, Union, Tuple


ActivationFunction = Callable[[np.ndarray], np.ndarray] # Function that can be considered as an activation function

class Layer:
    """
    One layer in a neural network
    """
    def __init__(self, size : int, activation_function : ActivationFunction):
        self.size = size
        self.activation_function = activation_function
        # Field initialization for later use
        self.neurons = np.zeros(size)

    def activate(self, raw_input : np.ndarray) -> None:
        """
        Store the input in the layer after applying the activation function on it
        """
        if len(raw_input) != self.size:
            raise ValueError(f"Input size {len(raw_input)} does not match layer size {self.size}")
        # Activation function are designed for the whole layers, rather than element by element
        self.neurons = self.activation_function(raw_input)

    def clear(self):
        """
        Clear the layer from any previous input, for further use
        """
        self.neurons = np.zeros(self.size)


class WeightMatrix:
    """
    Handles weights between 2 layers
    """
    def __init__(self, source_layer : Layer, target_layer : Layer):
        # Initialize the weights with random float values between -1 and 1
        self.weights = np.random.uniform(-1, 1, (source_layer.size, target_layer.size))

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Union[np.ndarray, float]:
        """
        Get all the weights for a specific neuron in layer 1, or the specific weight between 2 neurons (one from layer1 and the second layer2)
        For index type int - returns np.ndarray of weights
        For index type tuple - returns a float number
        """
        if isinstance(index, int):
            # if index is listed as an integer, all weights for a specific Neuron will be returned in layer1, e.g object[index1]
            # Also supports chained indexing, e.g object[index1][index2], here is how:
            # Matrix[index1] will return the np.ndarray, and then the [index2] will let numpy handle indexing inside the ndarray
            return self.weights[index]

        # if complex index is listed as a tuple, e.g object[index1, index2]
        elif isinstance(index, tuple):
            # Avoiding TypeError from len() if index is not a tuple, and checks that all elements are valid indexes
            if len(index) == 2 and all(isinstance(i, int) for i in index):
                # Looks directly in the ndarray for the specific weight
                return self.weights[index]
            raise IndexError("Index must be a tuple of 2 ints")

        else:
            raise IndexError("Index must be an int, or tuple of 2 ints")

    def __setitem__(self, index, value):
        """
        Setting new values in the weight matrix, the new value should be a numeric value (float or int).
        Sets new value for the weight between neuron index1 in layer1 and neuron index2 in layer2

        Supports:
        - Object[index1, index2] = NewValue
        - Object[index1][index2] = NewValue
        """
        # Check value first
        if not isinstance(value, (float, int)):
            raise ValueError("Value must be a float or an int number")
        # In cases of [1][2] the __getitem__ will be called first, and then numpy will handle the new value assignment
        if isinstance(index, tuple):
            # Avoiding TypeError from len() if index is not a tuple, and checks that all elements are valid indexes
            if len(index) == 2 and all(isinstance(i, int) for i in index):
                # Looks directly in the ndarray for the specific weight
                self.weights[index] = value
            else:
                raise IndexError("Index must be a tuple of 2 ints")
        else:
            raise IndexError("Index must be a tuple of 2 ints")
            # in case of single int, __getitem__ will be called instead


class NeuralNetwork:
    """
    Handles multiple layers and interconneweightscts them using multiple WeightMatrix objects
    """
    # TODO: learning rate annealing - Decrease the learning rate as the training progresses
    def __init__(self, layer_sizes : list[int], layer_types : list[ActivationFunction]):
        # TODO: Chenge to if - raise statement
        assert len(layer_sizes) == len(layer_types), "Layer sizes and types must be the same length"
        # TODO
        pass


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