import numpy as np

from neuralNerworkObjectBased import NeuralNetwork,ConnectedLayer,ActivationLayer

# training data
X = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
Y = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
nn = NeuralNetwork()
nn.addLayer(ConnectedLayer(2, 3))
nn.addLayer(ActivationLayer())
nn.addLayer(ConnectedLayer(3, 1))
nn.addLayer(ActivationLayer())

nn.fit(X, Y, epochs=1000, learningRate=0.1)

# test
out = nn.predict(X)
print(out)