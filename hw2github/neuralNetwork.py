import random
import numpy as np

class NeuralNetwork:
    def __init__(self,x0,hiddenLayerSpecs):
        self.x0 = x0 #input 
        self.hiddenLayerSpecs = hiddenLayerSpecs #[10,11,13,25,....] number of neurons per layer
        self.weights = []
        self.biases = []

        self.v_s = []

        for layerIndex,layerSpec in enumerate(self.hiddenLayerSpecs):
            if layerIndex == 0:
                self.weights.append([[random.uniform(-1.0,1.0) for n in range(len(x0))] for m in range(layerSpec)])
                self.biases.append([random.uniform(-1.0,1.0) for n in range(layerSpec)])
            else:
                self.weights.append([[random.uniform(-1.0,1.0) for n in range(self.hiddenLayerSpecs[layerIndex-1])] for m in range(layerSpec)])
                self.biases.append([random.uniform(-1.0,1.0) for n in range(layerSpec)])

        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

    def evaluate(self,x0):
        self.v_s = []
        for layerIndex in range(len(self.hiddenLayerSpecs)):
            if layerIndex == 0:
                self.v_s.append(self.sigmoid(np.array([np.dot(x0,np.array(self.weights[0][n]).T) for n in range(len(self.weights[0]))]) + np.array(self.biases[0]))) 
            else : 
                self.v_s.append(self.sigmoid(np.array([np.dot(self.v_s[layerIndex-1],np.array(self.weights[layerIndex][n]).T) for n in range(len(self.weights[layerIndex]))]) + np.array(self.biases[layerIndex]))) 
        return self.v_s[-1]/np.sum(self.v_s[-1])


    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))

x0 = np.array([0.1,0.11,1.3])

nn = NeuralNetwork(x0,[5,5,2])
print("start")
out = nn.evaluate(x0)
print(nn.v_s)
'''
neuralNetwork.weights[0][0] = [0.122,-1.22,0.33]
neuralNetwork.weights[0][1] = [0.06,1.36,2.11]
neuralNetwork.biases[0][0] = -1.123
neuralNetwork.biases[0][1] = 3.88

neuralNetwork.weights[1][0] = [0.187,-1.1]
neuralNetwork.weights[1][1] = [-0.187,1.1]
neuralNetwork.biases[1][0] = 0.36
neuralNetwork.biases[1][1] = -0.36
'''
#print(f"Weights:{neuralNetwork.weights}")
#print(f"Biases:{neuralNetwork.biases}")

