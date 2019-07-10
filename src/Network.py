import numpy as np
import random
import Neuron

class Network:
    
    def __init__(self, size, layers = ['sigmoid']):
        self.layers_num = len(size)
        self.size = size
        self.neuron = self.Create_Neurons(layers)
        #Each element k in the array is a matrix for the weights for the kth layer
        #Each row i the matrix is the weight from all the neurons in the kth layer to the ith neuron in the k+1 layer
        self.weight = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
        #Each element i in the array is an array of all the bias in the (i+1)th layer
        #the input layer has no biases
        self.bias = [np.random.randn(y, 1) for y in size[1:]]

    def Create_Neurons(self, layer_list):
        layer = []
        for i in range(self.layers_num):
            if(layer_list[i] == "sigmoid"):
                layer.append(Neuron.Sigmoid_Neuron)
        
        return layer

    #feed the input data a through the network and returns the prediction
    def Feed_Forward(self, a):
        for i in range(self.layers_num):
            a = self.neuron[i].output(a, self.weight[i], self.bias)
        
        return a

    def GSD():

    

    