"""
The goal is to be able to create any size - any purpose neural networks out of having only the list of layer sizes (e.g [8, 4, 2]) and layer types (activation function for each layer).

Usage example:
NeuralNetwork([784, 30, 20, 15, 10], [ai.Linear(), ai.Sigmoid(), ai.ReLU() ,ai.BinaryStep(), ai.Softmax()])
# Note: Custom activation functions can be used as well, as long as they are callable, accept a numpy array as input, and return a numpy array as output. Follow the ActivationFunction class to avoid conflicts.

Offered Classes:
Layer(size, activation_function)
WeightMatrix(layer1, layer2)
NeuralNetwork(layer_sizes, layer_types)

Offered Activation Functions:
Linear(values)
BinaryStep(values)
Sigmoid(values)
Softmax(values)
ReLU(values)
"""

import numpy as np
from typing import Callable, Union, Tuple
from abc import ABC, abstractmethod
from random import shuffle


# TODO: return type annotations
# TODO: and "Parameters:" section to all docstrings

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
        # Initialize the bias from source layer to target layer, as a 1D array with random values
        self.biases = np.random.uniform(-1, 1, (target_layer.size, ))

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

        # Iniialize backpropagation variables in advance
        self._activations = []  # Store activations for each layer during forward pass
        self._zs = []  # Store the pre-activation values (z) for each layer during forward pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network and store intermediate results
        """
        # Check that the input data is a 1D array and has the same size as the first layer
        validate_array(input_data, self.layers[0].size)

        # Reset last activations and zs
        self._activations = [input_data.copy()] # Store the very first input as "pre-layer" activation, in case first layer isn't linear. Also the input comes from outside the network, so for safety it is copied.
        self._zs = []  # Reset the pre-activation values

        # Pass the input data through the first layer, in case of a custom activation function
        current_activation = self.layers[0].activate(input_data)
        for i in range(len(self.layers)-1):
            # Get weights between layer i and i+1
            weights = self.weights[i]

            # Calculate the pre-activation values (z) for the next layer, including biases
            z = np.dot(current_activation, weights.weights) + weights.biases
            self._zs.append(z) 

            # Apply activation function for the next layer
            current_activation = self.layers[i+1].activate(z)
            self._activations.append(current_activation)

        # Return the final output
        return current_activation

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """
        Call the forward function using a simplified syntax
        """
        return self.forward(input_data)

    def _backpropagation(self, single_input_data: np.ndarray, expected_output: np.ndarray):
        """
        Calculates the gradients for the weights and biases using backpropagation algorithm for a single training example. Assumes all inputs are valid, and is not intended to be called directly by the user
        """
        # Forward pass to get the output, and fill the _activations and _zs lists
        output = self.forward(single_input_data)

        # Prepare empty gradient list, matching each weight matrix shape
        nabla_w = [np.zeros_like(w.weights) for w in self.weights]
        nabla_b = [np.zeros_like(w.biases) for w in self.weights]

        # Compute the error / loss / delta for the output layer
        # delta_L = (a_L - y) * f'(z_L)
        error = output - expected_output
        deriv = self.layers[-1].activation_function.derivative(self._zs[-1])
        delta = error * deriv

        # Gradient for the last layer's biases
        nabla_b[-1] = delta.copy()

        # Gradient for the last layer's weights
        prev_activation = self._activations[-2]
        nabla_w[-1] = np.outer(prev_activation, delta)

        # Backpropagate through the hidden layers l = L-1, ... 1
        # delta_l = (W_{l+1} delta_{l+1}) * sigma'(z_l) // Chain rule
        for layer_idx in range(2, len(self.layers)):
            z_val = self._zs[-layer_idx]  # pre-activation at layer l
            sigma_prime = self.layers[-layer_idx].activation_function.derivative(z_val)

            # propagate delta backward, weight matrix indexing offset by +1
            weight_next = self.weights[-layer_idx + 1].weights
            delta = np.dot(weight_next, delta) * sigma_prime

            # Gradient for this layer's biases - just the delta
            nabla_b[-layer_idx] = delta.copy()

            # gradient for this layer's weights: a_{l-1} outer delta_l
            prev_activation = self._activations[-layer_idx - 1]
            nabla_w[-layer_idx] = np.outer(prev_activation, delta)

        # Return list of gradient matrices for each weight matrix
        return nabla_w, nabla_b

    def gradient_descent_step(self, input_datas: np.ndarray, expected_outputs: np.ndarray,
                              learning_rate: float=0.7):
        # TODO: Validate inputs
        # Determine the number of training examples
        n_samples = input_datas.shape[0]
        
        # Initialize the accumulated gradients for each weight matrix
        sum_nabla_w = [np.zeros_like(w.weights) for w in self.weights]
        sum_nabla_b = [np.zeros_like(w.biases) for w in self.weights]

        # Loop over each training example, to compute individual gradients and later average them all
        for sample_input, sample_expected_output in zip(input_datas, expected_outputs):
            # Backpropagate this single example to get gradients for each weight matrix
            sample_gradients_w, sample_gradients_b = self._backpropagation(sample_input, sample_expected_output)
            # Accumulate gradients: add each sample's gradients to the running total
            sum_nabla_w = [total_grad + grad for total_grad, grad in zip(sum_nabla_w, sample_gradients_w)]
            sum_nabla_b = [total_grad + grad for total_grad, grad in zip(sum_nabla_b, sample_gradients_b)]

        # Update weights using averaged gradients 
        for i, w_matrix in enumerate(self.weights):
            # Compute the average gradient and scale by learning rate, then subtract from weights
            weight_update = (learning_rate / n_samples) * sum_nabla_w[i]
            w_matrix.weights -= weight_update

            # Bias update similarly
            bias_update = (learning_rate / n_samples) * sum_nabla_b[i]
            w_matrix.biases -= bias_update

    def train(self, data: dict, epochs: int, batch_size: int=128, learning_rate: float=0.9, annealing_factor: float=0.9):
        """
        Trains the Neural Network on the given data using mini-batch gradient descent, and temperature-like learning rate annealing.
        """

        # Separate data from dict
        inputs = []
        labels = []
        
        for image, label in data.items():
            validate_array(image, 28)
            validate_array(image[0], 28)

            # Flatten 28x28 to 784x
            inputs.append(image.flatten())
            # Convert label to one-hot list, to match output layer expected behaviour 
            one_hot = np.zeros(10)
            one_hot[label] = 1.0
            labels.append(one_hot)
        
        # Combine inputs and labels for easy shufling and batching
        combined = list(zip(inputs, labels))

        # train loop
        for i in range(epochs):
            shuffle(combined)
            
            # Create batches
            batches = [
                combined[i:i + batch_size:] for i in range(0, len(combined), batch_size)
            ]
            
            # Train on each batch
            for batch in batches:
                inputs = [pair[0] for pair in batch]
                labels = [pair[1] for pair in batch]

                self.gradient_descent_step(inputs, labels, learning_rate)

            learning_rate *= annealing_factor



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
        # Softmax derivative would require a Jacobian matrix and branching of the whole NeuralNetwork class training, so I used an aproximation
        s = self(input_data)
        return s * (1 - s)


class ReLU(ActivationFunction):
    """
    ReLU activation function
    Returns the input if positive, else returns 0
    """

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, input_data)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        return np.where(input_data > 0, 1, 0)