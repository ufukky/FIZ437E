# Creating J0 with noise
```
lenOfx = 50000
legend = []
X = []
for k in range(lenOfx):
    X.append(8/lenOfx * k)
X = np.array(X)
J0 = []
for x in X:
    if x == 0:
        J0.append(1)
    else:
        J0.append(np.sin(x)/x)
J0 = np.array(J0)

Y = []
for x in X:
    if x == 0:
        Y.append(1 + np.random.normal(0,1) * 0.1)
    else:
        Y.append((np.sin(x)/x) + np.random.normal(0,1) * 0.1)
Y = np.array(Y)

plt.plot(X,J0)
legend.append("J0")
plt.plot(X,Y,linewidth = 0.1,alpha=0.2)
legend.append("J0 (Noisy)")
```
# Sampling Random points for test and training
```
trainSampleCount = int(samplePointCount * 0.8)
trainIndexs = np.sort(np.random.randint(0,lenOfx,trainSampleCount)) 
trainX = np.array(X[trainIndexs])
trainY = np.array(Y[trainIndexs])
plt.scatter(trainX,trainY,s=[10 for n in range(len(trainIndexs))])
legend.append("Training Samples")
print(trainX)
print(trainY)

testSampleCount = 10
testIndexs = np.sort(np.random.randint(0,lenOfx,testSampleCount))
testX = np.array(X[testIndexs])
testY = np.array(Y[testIndexs])
plt.scatter(testX,testY,s=[1 for n in range(len(testIndexs))])
legend.append("Test Samples")
```
# Polynomial Regression
```
#------------------------------------ Polynomial Regression ----------------------------------------------
trainXreshaped = trainX.reshape(-1,1)
trainYreshaped = trainY.reshape(-1,1)

poly = PolynomialFeatures(degree=degree)
xPolyTrain = poly.fit_transform(trainXreshaped)
poly.fit(xPolyTrain,trainYreshaped)
linReg = LinearRegression()
linReg.fit(xPolyTrain,trainYreshaped)   
yPredCont = linReg.predict(poly.fit_transform(X.reshape(-1,1)))

plt.plot(X,yPredCont)
legend.append("Predicted Polynomial")
```
# Calculating L2 Loss and Root Mean Squared Error for training and validation set
```
trainingError = 0
trainingLoss = 0
for trainIndex in trainIndexs:
    yJ0 = J0[trainIndex]
    yPred = yPredCont[trainIndex]
    trainingError += ((yJ0 -yPred)**2)/trainSampleCount
    trainingLoss += (yJ0 -yPred)**2
trainingError = trainingError**(0.5)

testError = 0
testLoss = 0
for testIndex in testIndexs:
    yJ0 = Y[testIndex]
    yPred = yPredCont[testIndex]
    testError += ((yJ0 -yPred)**2)/testSampleCount
    testLoss += (yJ0 - yPred)**2
testError = testError**(0.5)

return trainingError,testError,trainingLoss,testLoss
```

