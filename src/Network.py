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
            if(len(layer_list) != 1):
                if(layer_list[i] == "sigmoid"):
                    n = Neuron.Sigmoid_Neuron()
                    layer.append(n)
            else:
                n = Neuron.Sigmoid_Neuron()
                layer.append(n)
        
        return layer

    #feed the input data a through the network and returns the prediction
    def Feed_Forward(self, a):
        for n, w, b in zip(self.neuron, self.weight, self.bias):
            a = n.Forward(a, w, b)
        
        return a

    def SGD(self, train_data, epochs, learning_rate, mini_batch_size, test_data = 0):
        n = len(train_data)
        if(test_data):
            n_test = len(test_data)

        for i in range(epochs):
            mini_batch = [train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for batch in mini_batch:
                nabla_b = [np.zeros(b.shape) for b in self.bias]
                nabla_w = [np.zeros(w.shape) for w in self.weight]
                #x is the input data, y is the expected output
                for x, y in batch:
                    #gets the gradient for the specific training data
                    delta_nb, delta_nw = self.Backprop(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nb)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nw)]

                self.bias = [b - (learning_rate/len(mini_batch)) * nb for b, nb in zip(self.bias, nabla_b)]
                self.weight = [w - (learning_rate/len(mini_batch)) * nw for w, nw in zip(self.weight, nabla_w)]
        
            if(test_data):
                print("epoch %d compeleted: %d / %d." % (i, self.validate(test_data), n_test))
            else:
                print("epoch %d compeleted." % i)
    
    def Backprop(self, x, y):

        nb = [np.zeros(b.shape) for b in self.bias]
        nw = [np.zeros(w.shape) for w in self.weight]

        a = [x]
        zs = []

        for n, w, b in zip(self.neuron, self.weight, self.bias):
            z = np.dot(w, a[-1]) + b
            zs.append(z)
            a.append(n.Activation(z))

        delta = (a[-1] - y) * self.neuron[-1].Differentiate(zs[-1])
        nb[-1] = delta
        nw[-1] = np.dot(delta, a[-2].transpose())

        for i in range(2, self.layers_num):
            delta = np.dot(self.weight[-i + 1].transpose(), delta) * self.neuron[-i].Differentiate(zs[-i])
            nb[-i] = delta
            nw[-i] = np.dot(delta, a[-(i + 1)].transpose())


        return (nb, nw)
    
    def validate(self, test_data):
        suc = 0
        for nx, ny in test_data:
          re = self.feedforward(nx)
          out = np.unravel_index(re.argmax(), re.shape)
          if(out[0] == ny):
            suc += 1
        
        return suc

