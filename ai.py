"""
The goal is to be able to create any size - any purpose neural networks out of having only the list of layer sizes (e.g [8, 4, 2]) and layer types (activation function for each layer).

Usage example:
NeuralNetwork([784, 100, 20, 15, 10], [Linear, ai.Sigmoid, ai.ReLU ,ai.BinaryStep, ai.Softmax])
# Note: Custom activation functions can be used as well, as long as they are callable, accept a numpy array as input, and return a numpy array as output. Follow the ActivationFunction class to avoid conflicts.

Offered Classes:
Layer(size, activation_function)
WeightMatrix(layer1, layer2)
NeuralNetwork(layer_sizes, layer_types)

Offered Functions:
Linear(values)
BinaryStep(values)
Sigmoid(values)
Softmax(values)
ReLU(values)
"""

import numpy as np
from typing import Callable, Union, Tuple
from abc import ABC, abstractmethod


# TODO: return type annotations

class ActivationFunction(ABC):
    """
    Abstract base class for activation functions that enforces implementation of both the function and its derivative
    """
    @abstractmethod
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Activation function should have the __call__ method implemented")

    @abstractmethod
    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Activation function should have the derivative method implemented")

class Layer:
    """
    One layer in a neural network
    """
    def __init__(self, size : int, activation_function : ActivationFunction):
        self.size = size
        self.activation_function = activation_function
        # Field initialization for later use
        self.neurons = np.zeros(size)

    def activate(self, input_data : np.ndarray) -> np.ndarray:
        """
        Store the input in the layer after applying the activation function on it
        Return the neurons after activation
        """
        validate_array(input_data, self.size)
        # Activation function are designed for the whole layers, rather than element by element
        self.neurons = self.activation_function(input_data)
        return self.neurons

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
        Get all the weights for a specific neuron from source_layer, or the specific weight between 2 neurons (one from source_layer and the second from target_layer)
        For index type int - returns np.ndarray of weights
        For index type tuple - returns a float number
        """
        if isinstance(index, int):
            # if index is listed as a single integer, all weights from a specific neuron in target_layer will be returned, e.g object[index1]
            # Also supports chained indexing, e.g object[index1][index2], here is how:
            # Matrix[index1] will return the np.ndarray, and then the [index2] will let numpy handle indexing inside the ndarray
            return self.weights[index]

        # if complex index is listed as a tuple, e.g object[index1, index2]
        elif isinstance(index, tuple):
            # Avoiding TypeError from len() if index is not a tuple, and then checks that all elements are valid indexes
            if len(index) == 2 and all(isinstance(i, int) for i in index):
                # Looks directly in the ndarray for the specific weight
                return self.weights[index]
            raise IndexError("Index must be a tuple of 2 ints")

        else:
            raise IndexError("Index must be an int, or tuple of 2 ints")

    def __setitem__(self, index, value):
        """
        Setting new values in the weight matrix, the new value should be a numeric value (float or int), or an instance of ndarray matching the expected structure (size, and numbers inside).
        Sets new value for the weight between neuron index1 in layer1 and neuron index2 in layer2
        Or sets new value for all the weights between neuron index1 in layer1 and all the neurons in layer2

        Supports:
        - Object[index] = ndarray of new values
        - Object[index1, index2] = NewValue
        - Object[index1][index2] = NewValue
        """
        # Check that the value type is valid first
        if not isinstance(value, (float, int, np.floating, np.integer, np.ndarray)):
            raise ValueError("Value must be a float or an int number")
        # If it's a NumPy array, validate its structure, if its a number it should not be validated
        if isinstance(value, np.ndarray):
            # Check that the array is 1D and has the same size as the target layer
            validate_array(value, self.weights.shape[1])

        # In cases of [1][2] the __getitem__ will be called first, and then numpy will handle the new value assignment

        # Handling the case of a complex index, e.g object[index1, index2]
        if isinstance(index, tuple):
            # Avoiding TypeError from len() if index is not a tuple, and checks that all elements are valid indexes
            if len(index) == 2 and all(isinstance(i, int) for i in index):
                # Looks directly in the ndarray for the specific weight
                self.weights[index] = value
            else:
                raise IndexError(
                    "Tuple index must be of length 2 and contain only integers: (int, int). "
                    f"Got: {index} (length {len(index)})"
                )

        # Handling the case of a single index, e.g object[index1]
        elif isinstance(index, int):
            # Numpy will handle invalid indexing, so I have to check only that the new value is a valid array
            # If the index is an integer then the value should be an array anyway, so its okay that the function is checking for it
            validate_array(value, self.weights.shape[1])
            self.weights[index] = value

        else:
            raise IndexError(
                "Index must be one of the following:\n"
                "- A single integer, with the new value being a 1D NumPy array of correct shape\n"
                "- A tuple of two integers (i, j) for assigning a specific weight\n"
                "- A chained index like [i][j]"
                )
            # in case of single int, __getitem__ will be called instead, so there is no need to handle the error message here


class NeuralNetwork:
    """
    Handles multiple layers and interconnects them using multiple WeightMatrix objects
    """
    # TODO: learning rate annealing - Decrease the learning rate as the training progresses
    def __init__(self, layer_sizes : list[int], activation_functions : list[ActivationFunction]):
        # Check the inputs are valid
        if len(layer_sizes) != len(activation_functions):
            raise ValueError("Layer sizes and types must be the same length")
        # Check sizes are valid
        if not all(isinstance(i, int) for i in layer_sizes):
            raise ValueError("Layer sizes must be a list of integers")
        # There is no way to check that the activation function is valid (Other than type annotations), so I will just assume that the user knows what they are doing
        
        # Create the layers
        self.layers = [Layer(layer_sizes[i], activation_functions[i]) for i in range(len(layer_sizes))]

        # Create the weight matrices; Specific weight is going to be accessed by the index of the source layer
        # e.g the weight matrix between layer 1 ([0] in self.layers) and layer 2 will be accessed by layers[0]
        self.weights = []
        # Too complex in one line format - hard to read
        for i in range(len(layer_sizes) - 1):
            # Create the weight matrix between layer i and layer i+1
            self.weights.append(WeightMatrix(self.layers[i], self.layers[i + 1]))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network
        """
        # Check that the input data is a 1D array and has the same size as the first layer
        validate_array(input_data, self.layers[0].size)
        # Pass the input data through the first layer, in case of a custom activation function
        input_data = self.layers[0].activate(input_data)
        for i in range(len(self.layers)-1):
            # Get weights between layer i and i+1
            weights = self.weights[i]
            # Calculate the input for the next layer
            input_data = self.layers[i+1].activate(np.dot(input_data, weights.weights))
        return input_data

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """
        Call the forward function using a simplified syntax
        """
        return self.forward(input_data)

            
def validate_array(array: np.ndarray, expected_size: int) -> None:
    """
    Check that the array is a 1D array and has the same size as is expected to be
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    if array.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array")
    if array.shape[0] != expected_size:
        raise ValueError(
            f"Input size {array.shape[0]} does not match expected size {expected_size}")


class Linear(ActivationFunction):
    """
    Linear activation function
    Returns the input as is
    """

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return input_data

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        # Derivative of linear function is always 1
        return np.ones(input_data.shape)


class BinaryStep(ActivationFunction):
    """
    Binary step activation function
    if number is positive, return 1, else return 0
    """

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return np.where(input_data > 0, 1, 0)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        # Derivative of binary step is always 0, and usually not used in practice, so Sigmoid derivative is used instead for approximation
        activated = self(input_data)
        return activated * (1 - activated)


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function
    returns a number between 0 and 1 for any number
    """

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        bounded = np.clip(input_data, -500, 500)
        return 1 / (1 + np.exp(-bounded))

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        activated = self(input_data)
        return activated * (1 - activated)


class Softmax(ActivationFunction):
    """
    Softmax activation function
    Converts a vector of numbers into a probability distribution
    """

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        exp_x = np.exp(input_data - np.max(input_data))
        return exp_x / np.sum(exp_x, axis=0)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        # Softmax derivative would require a Jacobian matrix and branching of the whole NeuralNetwork class training, so I used the Sigmoid derivative instead for approximation
        activated = self(input_data)
        return activated * (1 - self(activated))


class ReLU(ActivationFunction):
    """
    ReLU activation function
    Returns the input if positive, else returns 0
    """

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, input_data)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        return np.where(input_data > 0, 1, 0)