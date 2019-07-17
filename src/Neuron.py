import numpy as np

class Neurons:
    def Forward(input, weight, bias):
        pass
    
    def Activation(self, a):
        pass

class Sigmoid_Neuron(Neurons):
    def __init__(self):
        pass

    def Forward(a, weight, bias):
        z = np.dot(weight, a) + bias

        return 1.0/(1.0 + np.exp(-z))
    
    def Activation(self, a):
        return 1.0/(1.0 + np.exp(-z))
