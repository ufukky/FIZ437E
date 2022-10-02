#Ufuk Köksal Yücedağ 090180117
import cv2
import numpy as np
import os
from operator import itemgetter

from setuptools import SetuptoolsDeprecationWarning

def k_NNMain(testImageDir,epochLimit = 500): #input: testImage output:frog/bird
    frogsTrainDataDirectory = './data/train/frogs' #file directory for traning data of frogs
    birdsTrainDataDirectory = './data/train/birds' #file directory for traning data of birds
    testImage = cv2.imread(testImageDir,0)         #loading 64x64 pixel greyscale image to 64x64 matrix with values from 0 to 255
    testImageVec = testImage.ravel()               #turning 64x64 matrix to 1x4096 vector
    leastKNeighbours= []                           #[{label: <bird or frog> , fileName: <fileName> , distance: <eulerDistance>'}]

    _epochLimitFrogs = 0
    for frogFile in os.listdir(frogsTrainDataDirectory):        #calculate test image nn-algorithm for frogs via iterating through 450 training images
        if(_epochLimitFrogs > epochLimit):
            break
        file = os.path.join(frogsTrainDataDirectory, frogFile)
        if os.path.isfile(file):                                #check for file's existance
            frogTrainImage = cv2.imread(file,0)                 #load it to 64x64 matrix 
            frogTrainImageVec = frogTrainImage.ravel()          #64x64 matrix -> 1x4096 vector
            #evaluateNN(leastKNeighbours, testImageVec=testImageVec , trainImageVec=frogTrainImageVec, label="frog", fileName=file) #run nn-algorithm
            leastKNeighbours.append(evaluateNN(leastKNeighbours, testImageVec=testImageVec , trainImageVec=frogTrainImageVec, label="frog", fileName=file))
        _epochLimitFrogs = _epochLimitFrogs + 1

    _epochLimitBirds = 0
    for birdFile in os.listdir(birdsTrainDataDirectory):        #calculate test image nn-algorithm for frogs via iterating through 450 training images
        if(_epochLimitBirds > epochLimit):
            break
        file = os.path.join(birdsTrainDataDirectory, birdFile)
        if os.path.isfile(file):                                #check for file's existance
            birdTrainImage = cv2.imread(file,0)                 #load it to 64x64 matrix 
            birdTrainImageVec = birdTrainImage.ravel() #64x64 -> 1x4096
            #evaluateNN(leastKNeighbours, testImageVec=testImageVec , trainImageVec=birdTrainImageVec, label="bird", fileName=file) #run nn-algorithm
            leastKNeighbours.append(evaluateNN(leastKNeighbours, testImageVec=testImageVec , trainImageVec=birdTrainImageVec, label="bird", fileName=file))
        _epochLimitBirds = _epochLimitBirds + 1
    
    leastKNeighbours = sorted(leastKNeighbours, key=itemgetter('distance'))

    resultsForKs = [] #storing guessed labels for input image with k=1,3,5,7,9 at the same epoch for faster calculation time.
    for l in range(0,5):
        k=2*l+1

        _leastKNeighbours = leastKNeighbours[0:k]                    #limiting the closest neigbors array via getting only closest k neigbours

        numberOfFrogNeigbours = 0                                   #final evaluation of bird data with majorty vote methode
        numberOfBirdNeigbours = 0
        for leastNeighbour in _leastKNeighbours: 
            if(leastNeighbour['label'] == 'frog'):
                numberOfFrogNeigbours = numberOfFrogNeigbours + 1
            elif(leastNeighbour['label'] == 'bird'):
                numberOfBirdNeigbours = numberOfBirdNeigbours + 1

        if(numberOfBirdNeigbours > numberOfFrogNeigbours):
            resultsForKs.append("bird")
        else:
            resultsForKs.append("frog")
    return resultsForKs

def evaluateNN(kNArray,trainImageVec,testImageVec,label,fileName):
    dist = np.linalg.norm(trainImageVec - testImageVec) #euler distance between test and training data
    return {
            "label":label,                              
            "fileName": fileName,
            "distance": dist
    }



#-------------------------------------------------------- Main Start -----------------------------------------------------------
#testing accuracy of knn with training data with constant epoch limit = 450 and diffrent k values

birdsTrainingDataDirectory = './data/train/birds'
frogsTrainingDataDirectory = './data/train/frogs'

epochLimit = 450        #all data
print(f"epoch:{epochLimit}")
with open('results.txt', 'a') as f:
    f.write(f'epochLimit:{epochLimit}\n')

testResults = [[],[],[],[],[]] # {guessedLabel : 'frog', realLabel : 'frog'}
for birdFile in os.listdir(birdsTrainingDataDirectory):        #test knn with all birds and frogs via iterating through 100 test images
    testFile = os.path.join(birdsTrainingDataDirectory, birdFile)
    if os.path.isfile(testFile):    
        arrayResult = k_NNMain(testFile,epochLimit) 
        for _k in range(0,5):
            testResults[_k].append({'guessedLabel' : arrayResult[_k], 'realLabel' : 'bird'})
for frogFİle in os.listdir(frogsTrainingDataDirectory):        #test knn with all birds and frogs via iterating through 100 test images
    testFile = os.path.join(frogsTrainingDataDirectory, frogFİle)
    if os.path.isfile(testFile):    
        arrayResult = k_NNMain(testFile,epochLimit)
        for _k in range(0,5):
            testResults[_k].append({'guessedLabel' : arrayResult[_k], 'realLabel' : 'frog'})

for _k in range(0,5): 
    rightGuessCount = 0         #storing number of right and wrong guesses for calculating accuracy for each value for k
    wrongGuessCount = 0
    for testResult in testResults[_k]:
        if(testResult['guessedLabel'] == testResult['realLabel']):
            rightGuessCount = rightGuessCount + 1
        else:
            wrongGuessCount = wrongGuessCount + 1

    print(f"k:{2*_k+1}  acc:{rightGuessCount/(rightGuessCount+wrongGuessCount)}")
    with open('results.txt', 'a') as f:
        f.write(f'k:{2*_k+1}  acc:{rightGuessCount/(rightGuessCount+wrongGuessCount)}\n')

with open('results.txt', 'a') as f:
    f.write(f'----------------------------------------------------------------------------\n')


#testing accuracy of knn with test-data with diffrent epoch limits and k values

birdsTestDataDirectory = './data/test/birds'
frogsTestDataDirectory = './data/test/frogs'

for _e in range(1,51):
    #epochLimit = 100
    epochLimit = _e*10
    print(f"epoch:{epochLimit}")
    with open('results.txt', 'a') as f:
        f.write(f'epochLimit:{epochLimit}\n')

    testResults = [[],[],[],[],[]] # {guessedLabel : 'frog', realLabel : 'frog'}
    for birdFile in os.listdir(birdsTestDataDirectory):        #test knn with all birds and frogs via iterating through 100 test images
        testFile = os.path.join(birdsTestDataDirectory, birdFile)
        if os.path.isfile(testFile):    
            arrayResult = k_NNMain(testFile,epochLimit)
            for _k in range(0,5):
                testResults[_k].append({'guessedLabel' : arrayResult[_k], 'realLabel' : 'bird'})
    for frogFİle in os.listdir(frogsTestDataDirectory):        #test knn with all birds and frogs via iterating through 100 test images
        testFile = os.path.join(frogsTestDataDirectory, frogFİle)
        if os.path.isfile(testFile):    
            arrayResult = k_NNMain(testFile,epochLimit)
            for _k in range(0,5):
                testResults[_k].append({'guessedLabel' : arrayResult[_k], 'realLabel' : 'frog'})

    for _k in range(0,5): 
        rightGuessCount = 0         #storing number of right and wrong guesses for calculating accuracy
        wrongGuessCount = 0
        for testResult in testResults[_k]:
            if(testResult['guessedLabel'] == testResult['realLabel']):
                rightGuessCount = rightGuessCount + 1
            else:
                wrongGuessCount = wrongGuessCount + 1

        print(f"k:{2*_k+1}  acc:{rightGuessCount/(rightGuessCount+wrongGuessCount)}")
        with open('results.txt', 'a') as f:
            f.write(f'k:{2*_k+1}  acc:{rightGuessCount/(rightGuessCount+wrongGuessCount)}\n')

    with open('results.txt', 'a') as f:
        f.write(f'----------------------------------------------------------------------------\n')
