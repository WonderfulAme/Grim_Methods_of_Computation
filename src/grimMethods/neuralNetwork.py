import torch
from grimMethods.neuron import Neuron

class NeuralNetwork:
    def __init__(self, layer: list, layer_types: list, activations: list, loss_function: str = 'cross_entropy', optimizer: str = 'SGD', regularization: str = 'L2', learning_rate_scheduler: str = None):
        """
        Initializes a customizable neural network architecture.

        The neural network structure is defined by specifying the number of layers, their types, activation functions, 
        loss function, optimizer, regularization method, and an optional learning rate scheduler.

        Args:
            layer (list): List of integers specifying the number of units in each layer (e.g., [10, 1, 3]).
            layer_types (list): List of strings specifying the type of each layer. This can be 'input', 'dense', 'convolutional', 'recurrent', 'pooling', 'dropout', 'output'.
            activations (list): List of strings specifying the activation function for each layer. This can be 'relu', 'sigmoid', 'softmax', 'tanh'.
            loss_function (str, optional): Loss function to use during training. Default is 'cross_entropy'. Other options can be 'mean_squared_error'.
            optimizer (str, optional): Optimization algorithm to use. Default is 'SGD'. Other options are 'Adam', 'RMSprop', 'AdamW'.
            regularization (str, optional): Regularization method to apply. Default is 'L2'. Other options can be 'L1', 'dropout'.
            learning_rate_scheduler (str, optional): Learning rate scheduler to use. Default is None.

        # The neural network structure is defined by the 'layer', 'layer_types', and 'activations' lists, 
        # allowing for flexible configuration of the number of layers, their types (input, hidden, output), 
        # and the activation functions used in each layer.
        """
        self.num_layers = len(layer)
        self.layers = layer
        self.layer_types = layer_types
        self.activations = activations
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.regularization = regularization
        self.learning_rate_scheduler = learning_rate_scheduler
        self.neurons = []

        for i in range(self.num_layers):
            layer_neurons = []
            for j in range(self.layers[i]):
                neuron = Neuron(activation=self.activations[i])
                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)
