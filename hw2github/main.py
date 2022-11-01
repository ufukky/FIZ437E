import numpy as np
import matplotlib.pyplot as plt
from logisticRegression import LogisticRegression
from supportVectorMachine import SupportVectorMachine,SVM

entries = []

dataFile = open('haberman.data', 'r') #read data to entries array
lines = dataFile.readlines()  
for line in lines:
    _entry = (line.split())[0].split(',')
    entry=[]
    for val in _entry:
        entry.append(float(val)) # turn string entries to floats
    entries.append(entry)
entries = np.array(entries)



entries[:,-1] = entries[:,-1] - 1 #output translated to 0 and 1 from 1 and 2
trainingSet = np.array(entries[0:int(len(entries) * 0.9)]) # 90% of the data reserved for training (275)entries
testSet = np.array(entries[int(len(entries) * 0.9):]) # 10% of data reserved for testing (31)

X = trainingSet[:,0:3] #inputs
Y = trainingSet[:,3:] #outputs
Xt = testSet[:,0:3] #inputs
Yt = testSet[:,3:] #outputs

#Logistic Regression
'''
logReg = LogisticRegression()
logReg.train(X,Y,10,1000,0.0001)

preds = logReg.predict(Xt)
err = 0
for i in range(len(preds)):
    p = preds[i]
    t = Yt[i]
    if(p != t):
        err += 1
accVal = 1 - (err/len(Yt))
accVal = round(accVal,4)
print(f"Validation Accuracy:{accVal}")

preds = logReg.predict(X)

err = 0
for i in range(len(preds)):
    p = preds[i]
    t = Y[i]
    if(p != t):
        err += 1
accTra = 1 - (err/len(X))
accTra = round(accTra,4)
print(f"Train Accuracy:{accTra}")


pltX = np.linspace(0,len(logReg.losses),len(logReg.losses))
pltY = np.array(logReg.losses)
plt.xlabel("#Epoch")
plt.ylabel("Loss")
plt.title("Train Loss Curve")
plt.plot(pltX,pltY)
plt.text(300,-0.06,f"Train Accuracy:{accTra}\nValidation Accuracy:{accVal}")
plt.show()
'''

#Support Vector Machine
#Y = Y*2 - 1 # 0->1 --> -1->1
tmp = np.array([Y[n][0] for n in range(len(Y))])
Y = tmp

#Yt = Yt*2 -1 
tmp = np.array([Yt[n][0] for n in range(len(Yt))])
Yt = tmp

'''
supVecM = SupportVectorMachine()
supVecM.train(X,Y,lbd=0.1,lr=0.000001,epochs=1000)
w,b,l = supVecM.w,supVecM.b,supVecM.losses
pltX = np.linspace(0,len(supVecM.losses),len(supVecM.losses))
pltY = np.array(supVecM.losses)
plt.plot(pltX,pltY)
plt.show()
'''

svm = SVM(lambda_param=0.01,n_iters=1000,learning_rate=0.00000001)
svm.fit(X,Y)
w,b,l =  svm.w,svm.b,svm.losses
preds = svm.predict(X)

faults = 0
for i in range(len(preds)):
    if preds[i] != Y[i]:
        faults += 1
acc = 1 - faults/len(X)
print(f"acc:{acc}")
print(f"w:{w},b:{b},l:{l}")

pltX = np.linspace(0,len(l),len(l))
pltY = np.array(l)
plt.plot(pltX,pltY)
plt.show()