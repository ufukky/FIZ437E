import time
import random
from tkinter import W
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self,x0,hiddenLayerSpecs):
        self.x0 = x0 #input 
        self.hiddenLayerSpecs = hiddenLayerSpecs #[10,11,13,25,....] number of neurons per layer
        self.weights = []
        self.biases = []

        self.layers_post_activation = []
        self.layers_pre_activation = []


        for layerIndex,layerSpec in enumerate(self.hiddenLayerSpecs):
            if layerIndex == 0:
                self.weights.append([[random.uniform(-1.0,1.0) for n in range(len(x0))] for m in range(layerSpec)])
                self.biases.append([random.uniform(-1.0,1.0) for n in range(layerSpec)])
            else:
                self.weights.append([[random.uniform(-1.0,1.0) for n in range(self.hiddenLayerSpecs[layerIndex-1])] for m in range(layerSpec)])
                self.biases.append([random.uniform(-1.0,1.0) for n in range(layerSpec)])

        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

    def fordwardPass(self,x0):
        self.layers_post_activation = []
        self.layers_pre_activation = []
        for layerIndex in range(len(self.hiddenLayerSpecs)):
            if layerIndex == 0:
                self.layers_pre_activation.append(np.array([np.dot(x0,np.array(self.weights[0][n]).T) for n in range(len(self.weights[0]))]) + np.array(self.biases[0])) 
                self.layers_post_activation.append(self.sigmoid(self.layers_pre_activation[layerIndex])) 
            else : 
                self.layers_pre_activation.append(np.array([np.dot(self.layers_post_activation[layerIndex-1],np.array(self.weights[layerIndex][n]).T) for n in range(len(self.weights[layerIndex]))]) + np.array(self.biases[layerIndex]))
                self.layers_post_activation.append(self.sigmoid(self.layers_pre_activation[layerIndex])) 
        output = self.layers_post_activation[-1]
        normalized_output = self.layers_post_activation[-1]/np.sum(self.layers_post_activation[-1])
        return output,normalized_output,self.layers_pre_activation,self.layers_post_activation

    def Error(self,y,yt):
        if y.shape != yt.shape:
            return -1
        element_vise_error = (1/2)*(y - yt)**2
        total_error = np.sum(element_vise_error)
        return element_vise_error,total_error
    
    def RandomizeWithMutationRate(self,mutationRate):
        for layerIndex in range(len(self.weights)):
            for neuronIndex in range(len(self.weights[layerIndex])):
                self.biases[layerIndex][neuronIndex] += np.random.normal(0, 1, 1)[0] * mutationRate
                for weightIndex in range(len(self.weights[layerIndex][neuronIndex])):
                    self.weights[layerIndex][neuronIndex][weightIndex] += np.random.normal(0, 1, 1)[0] * mutationRate
        return self

    def Backpropagate(self,yt,lr): #yt:true y, lr:learning rate
        new_weights = self.weights
        storedChanges = self.weights
        for layerIndex in reversed(range(len(self.hiddenLayerSpecs))):
            if layerIndex != len(self.hiddenLayerSpecs) - 1: #hidden layer 
                for hiddenNeuronIndex in range(self.hiddenLayerSpecs[layerIndex]):
                    y = self.sigmoid_prime(self.layers_post_activation[layerIndex][hiddenNeuronIndex])
                    for weightIndex in range(len(self.weights[layerIndex][hiddenNeuronIndex])):
                        weight = self.weights[layerIndex][hiddenNeuronIndex][weightIndex] #w7
                        z = self.layers_post_activation[layerIndex - 1][weightIndex] if layerIndex != 0 else self.x0[weightIndex]
                        x = 0
                        for forwardConnectedNeuronIndex in range(self.hiddenLayerSpecs[layerIndex+1]):
                            forwardWeight = self.weights[layerIndex+1][forwardConnectedNeuronIndex][hiddenNeuronIndex]
                            forwardXY = storedChanges[layerIndex+1][forwardConnectedNeuronIndex][hiddenNeuronIndex]
                            x += forwardXY[0]*forwardXY[1]*forwardWeight
                        gradient = x*y*z
                        storedChanges[layerIndex][hiddenNeuronIndex][weightIndex] = [x,y] 
                        new_weights[layerIndex][hiddenNeuronIndex][weightIndex] -= lr*gradient 
                        

            else:
                for outputNeuronIndex in range(self.hiddenLayerSpecs[layerIndex]):
                    outputNeuronWeights = self.weights[layerIndex][outputNeuronIndex]
                    outputNeuronBias = self.biases[layerIndex][outputNeuronIndex]
                    x = yt[outputNeuronIndex] - self.layers_post_activation[layerIndex][outputNeuronIndex]
                    y = self.sigmoid_prime(self.layers_post_activation[layerIndex][outputNeuronIndex])
                    
                    for connectedNeuronIndex in range(self.hiddenLayerSpecs[layerIndex - 1]):
                        weight = outputNeuronWeights[connectedNeuronIndex]
                        bias = outputNeuronBias
                        value = self.layers_post_activation[layerIndex - 1][connectedNeuronIndex]
                        z = value
                        gradient = x*y*z

                        storedChanges[layerIndex][outputNeuronIndex][connectedNeuronIndex] = [x,y] 
                        new_weights[layerIndex][outputNeuronIndex][connectedNeuronIndex] -= lr * gradient
                        

            
        self.weights = new_weights

    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))
    
    def sigmoid_prime(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
#-----------------------------------------------------------------------------------------------------------------------------------------

entries = []

dataFile = open('haberman.data', 'r') #read data to entries array
lines = dataFile.readlines()  
for line in lines:
    _entry = (line.split())[0].split(',')
    entry=[]
    for val in _entry:
        entry.append(float(val)) # turn string entries to floats
    entries.append(entry)



for entry in entries:#output translated to 0 and 1 from 1 and 2
    entry[-1] -= 1
trainingSet = entries[0:int(len(entries) * 0.9)] # 90% of the data reserved for training (275)entries
testSet = entries[int(len(entries) * 0.9):] # 10% of data reserved for testing (31)

X = np.array(trainingSet)[:,0:3] #inputs
Y = np.array(trainingSet)[:,3:] #outputs
Xt = np.array(testSet)[:,0:3] #inputs
Yt = np.array(testSet)[:,3:] #outputs

nGenerations = 1000
PopulationSize = 50
bestNN = NeuralNetwork(X[0],[5,5,1])
gens = []
errors = []

for generationCount in range(nGenerations):
    startTime = time.time()
    errorBest = 0
    for trainingIndex in range(len(X)):
        yBest = bestNN.fordwardPass(X[trainingIndex])[0]
        errorBest += bestNN.Error(yBest,Y[trainingIndex])[1]/float(len(X))
    generation = [bestNN.RandomizeWithMutationRate(mutationRate=0.001) for g in range(PopulationSize)]

    minAgent = None
    minAgentError = None
    for agent in generation:
        agentError = 0
        for trainingIndex in range(len(X)):
            y = agent.fordwardPass(X[trainingIndex])[0]
            agentError +=  agent.Error(y,Y[trainingIndex])[1]/float(len(X))
        if minAgentError == None or agentError <= minAgentError:
            minAgent = agent
            minAgentError = agentError
    if minAgentError <= errorBest:
        bestNN = minAgent
    gens.append(generationCount)
    errors.append(errorBest)

    remain = (time.time()-startTime) * (nGenerations-generationCount)
    if generationCount % 10 == 0:
        print(f"{generationCount}/{nGenerations} remain:{round(remain/60)} mins")


preds = []
for testSetIndex in range(len(X)):
    x = X[trainingIndex]
    preds.append(bestNN.fordwardPass(X[testSetIndex])[0])

for pred in preds:
    if pred > 0.5:
        pred = 1
    else:
        pred = 0
    

faults = 0
for i in range(len(preds)):
    if preds[i] != Y[i]:
        faults += 1
acc = 1 - faults/len(X)
print(f"acc:{acc}")


plt.plot(gens,errors)
plt.show()
