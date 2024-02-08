"""neural_net.py

Holds classes for neural network
"""
import math
import numpy as np
import operator

from numpy.random import random_sample
from time import time
from typing import Generator

def sigmoid(input:float) -> float:
    """Run sigmoid algorithm on a value"""
    return 1 / (1 + math.exp(-input))

def d_sigmoid(input:float) -> float:
    """Derivative of sigmoid algorithm"""
    return (sig := sigmoid(input)) * (1 - sig)

def total_cost(expected_outputs:list[float], predicted_outputs:list[float]) -> float:
    """Compute total cost of network"""
    expected = np.array(expected_outputs)
    predicted = np.array(predicted_outputs)
    return np.sum(expected * np.log(predicted) + (1 - expected) * (np.log(1 - predicted)))

class NeuralNet():
    """Class for neural network"""

    class Neuron():
        """Class for neurons in network"""
        weights: list[float] = []
        value: float = 0.5 # Value after sigmoid algorithm
        bias: float = 0.5

        def __init__(self, num_weights:int):
            self.weights = random_sample(size=num_weights)

    #
    # Info about network
    #
    _layers: list[int]
    _learn_rate: float
    _input_size: int
    _output_size: int

    #
    # Neurons in network
    #
    _neurons: list[list[Neuron]]

    def __init__(self, num_inputs:int, layers:list[int], num_outputs:int,
                 learn_rate:float):
        self._layers = layers
        self._learn_rate = learn_rate
        self._input_size = num_inputs
        self._output_size = num_outputs
        self._neurons = []

        prev_layer_size: int = num_inputs
        for layer_size in layers:
            self._neurons.append([self.Neuron(prev_layer_size) for _ in range(layer_size)])
            prev_layer_size = layer_size
        self._neurons.append([self.Neuron(prev_layer_size) for _ in range(num_outputs)])

    def propagate(self, inputs:list[float]) -> list[float]:
        """Propagate values through network"""
        prev_layer_values: list[float] = inputs
        for layer in self._neurons:
            for neuron in layer:
                neuron.value = sigmoid(np.dot(prev_layer_values, neuron.weights) + neuron.bias)
            prev_layer_values = self._neurons_to_values(layer)
        return prev_layer_values

    def back_propagate(self, inputs:list[float], expected_outputs:list[float]) -> float:
        """Backpropagation algorithm"""
        # Calculate cost and output layer error
        predicted_outputs = self._neuron_values_as_list(-1)
        cost = total_cost(expected_outputs, predicted_outputs)
        cur_layer_errors: list[float] = np.subtract(expected_outputs, predicted_outputs)

        for l_idx in range(len(self._neurons)-1, 0, -1):
            # Get values of previous layer for updating weights
            if l_idx != 0:
                new_errors = np.zeros(len(self._neurons[l_idx-1]))
                backward_layer_values = self._neuron_values_as_gen(l_idx-1)
            else:
                new_errors = np.zeros(self._input_size)
                backward_layer_values = (input for input in inputs)

            # Update weights for neurons in each layer
            for n_idx, neuron in enumerate(self._neurons[l_idx]):
                gradient: float = cur_layer_errors[n_idx] * d_sigmoid(neuron.value)
                for w_idx, weight in enumerate(neuron.weights):
                    neuron.weights[w_idx] += self._learn_rate * gradient * -cost * next(backward_layer_values)
                    new_errors[w_idx] += gradient * weight
                neuron.bias += self._learn_rate * gradient * -cost
            cur_layer_errors = new_errors
        return cost

    def _neuron_values_as_list(self, layer_idx:int) -> list[float]:
        """Get the values of the neurons as a list

        Note:
            `layer_idx` is used as array accessor, meaning -1,-2,etc. is valid
        """
        return [neuron.value for neuron in self._neurons[layer_idx]]

    def _neuron_values_as_gen(self, layer_idx:int) -> Generator[float, None, None]:
        """Get values of neurons at `layer_idx` as generator

        Used for optimization to reduce amount of lists created
        """
        for neuron in self._neurons[layer_idx]:
            yield neuron.value

    def _neurons_to_values(self, layer:list[Neuron]) -> list[float]:
        """Get values of neurons from list"""
        return [neuron.value for neuron in layer]

def train(operator) -> NeuralNet:
    """Test neural network training on a boolean operator"""
    # Init with 2 inputs, 1 hidden layer of size 4, and 1 output with 0.09 learning rate
    nn = NeuralNet(2, [ 4 ], 1, 0.09)
    try:
        # Train over 100,000 iterations of a boolean operation over two inputs
        start_time = time()
        for it in range(100_001):
            values = np.random.choice([1, 0], 2)            # Two random inputs
            guess = nn.propagate(values)                    # Get guessed output
            answer = [int(operator(values[0], values[1]))]  # Get correct output
            cost = nn.back_propagate(values, answer)        # Backpropagate and get error of network

            # Output learning statistics over training
            if it % 1000 == 0:
                print(f"It: {it:6} Input: {values} guess: [{guess[0]:.6f}] answer: {answer} cost: {cost:.6f}")
        end_time = time()
        print(f"Training Time: {end_time - start_time:.5f}s")
    # Handle keyboard interrupts because traceback is annoying
    except KeyboardInterrupt:
        print("Keyboard Interrupt: exiting...")
    # Return trained neural network
    return nn

def validate(nn:NeuralNet, operator) -> None:
    """Validate the test neural network"""
    # Generate validation set
    values: list[list[int]] = []
    for val_1 in range(2):
        for val_2 in range(2):
            values.append([val_1, val_2])

    # Run network on validation set
    for it, inputs in enumerate(values):
        guess = nn.propagate(inputs)
        answer = [int(operator(inputs[0], inputs[1]))]
        cost = total_cost(answer, guess)
        print(f"Validation: {it:3} Input: {inputs} guess: [{guess[0]:.6f}] answer: {answer} cost: {cost:.6f}")
    pass

if __name__ == "__main__":
    op = operator.or_
    nn = train(op)
    validate(nn, op)