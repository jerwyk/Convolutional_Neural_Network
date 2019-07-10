import numpy as np

class Neurons:
    def output(input, weight, bias):
        pass

class Sigmoid_Neuron(Neurons):
    def __init__(self):
        pass

    def output(input, weight, bias):
        z = np.dot(weight, input) + bias

        return 1.0/(1.0 + np.exp(-z))
