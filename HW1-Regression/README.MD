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
# Comparing training and validation error for 10 samples
```
#----------------------------------------------- 10 Samples and validation versus training error ------------------
'''
avgTrain = 0
avgTest = 0
for n in range(100):
    trainingError,testError,trainingLoss,testLoss = polyFit(samplePointCount=10)
    avgTest += testLoss/100
    avgTrain += trainingError/100
```
![alt text](https://github.com/ufukky/FIZ437E/blob/main/HW1-Regression/10sampleTrainVsTestErr.png?raw=true)

# L2 Loss for validation set versus sample points
```
train = []
test = []
sampleCounts = [10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100,1000,10000]
for samplePointCount in sampleCounts:
    avgTrain = 0
    avgTest = 0
    for n in range(100):
        trainingError,testError,trainingLoss,testLoss = polyFit(samplePointCount=samplePointCount)
        avgTest += testLoss/100
        avgTrain += trainingLoss/100
    train.append(avgTrain)
    test.append(avgTest)
    print(f"sc:{samplePointCount} | trainErr:{avgTrain} | testErr:{avgTest}")
```
![alt text](https://github.com/ufukky/FIZ437E/blob/main/HW1-Regression/nSampleVsL2LossVal.png?raw=true)

# Overfitted Curve Examples
![alt text](https://github.com/ufukky/FIZ437E/blob/main/HW1-Regression/overfitting.png?raw=true)

# Ridge regression for eliminating overfitting with various aplha values
```
xTrain = [0.8608, 1.752, 2.13616, 2.23888, 3.33984, 4.05392, 4.12464, 5.35536, 5.62784, 7.07184]
xTrain = np.array(xTrain)

yTrain = [0.7812559,0.714773,0.54731242,0.34609456,-0.03401954,-0.21559265,-0.37846568,-0.13502542,0.04664801,0.07457167]
yTrain = np.array(yTrain)
plt.scatter(xTrain,yTrain)
legend.append("Training Points")

trainXreshaped = xTrain.reshape(-1,1)
trainYreshaped = yTrain.reshape(-1,1)

poly = PolynomialFeatures(degree=8)
xPolyTrain = poly.fit_transform(trainXreshaped)
poly.fit(xPolyTrain,trainYreshaped)
linReg = LinearRegression()
linReg.fit(xPolyTrain,trainYreshaped)   
yPredCont = linReg.predict(poly.fit_transform(X.reshape(-1,1)))

#plt.plot(X,yPredCont,linewidth = 0.5,alpha=0.7)
#legend.append("Predicted Polynomial")

alphas = [0.0000002* n for n in range(0,6)]
for alpha in alphas:
    ridge = Ridge(alpha = alpha)
    ridge.fit(xPolyTrain,trainYreshaped)
    ridgePredCont = ridge.predict(poly.fit_transform(X.reshape(-1,1)))
    plt.plot(X,ridgePredCont,linewidth=0.7)
    if alpha == 0:
        legend.append(f'Overfitting (alpha:{alpha})')
    else:
        legend.append(f"Ridge Regression (alpha:{alpha})")
```
![alt text](https://github.com/ufukky/FIZ437E/blob/main/HW1-Regression/ridge.png?raw=true)
