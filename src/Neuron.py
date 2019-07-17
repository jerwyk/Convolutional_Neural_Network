import numpy as np

class Neurons:
    def Forward(self, input, weight, bias):
        pass
    
    def Activation(self, a):
        pass

class Sigmoid_Neuron(Neurons):
    def __init__(self):
        pass

    def Forward(self, a, weight, bias):
        a = np.dot(weight, a) + bias

        return self.Activation(a)
    
    def Activation(self, a):
        return 1.0/(1.0 + np.exp(-a))

    def Differentiate(self, a):
        s = self.Activation(a)
        return s * (1 - s)
