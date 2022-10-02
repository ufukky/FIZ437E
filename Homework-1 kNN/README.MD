# Data Preperation
## Data for birds
500 images of birds are downloaded from the first 500 URL's of ./data/bird_images.txt using httpRequest. Downloaded images of birds than converted to greyscale and resized to 64x64 - same size of frog images -
```
imageCount = 500 #downloading 500 images of birds from image urls in bird_images.txt

file1 = open('./data/bird_images.txt', 'r')
Lines = file1.readlines()

imageIndex = 0
while imageCount > 0:
    imageUrl = Lines[imageIndex].strip().split('\t')[0] #stripping imageUrl portion of text for http request
    print(imageUrl)
    print("Test {}-{}: {}".format(imageCount,imageIndex, imageUrl))

    try :
        resp = requests.get(imageUrl, stream=True).raw                      #http request for downloading the bird image
        image = np.asarray(bytearray(resp.read()), dtype="uint8")           #turning raw respose to width*height sized 2D numpy array
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)                       #turning numpy array to cv2 compatible array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                     #turning rgb image to greyscale
        image = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)  #resizing downloaded image to 64x64 
        cv2.imwrite('./data/train/birds/{}.jpg'.format(imageCount), image)  #saving prepared image to training images of birds
        imageCount = imageCount - 1
        imageIndex = imageIndex + 1
    except :
        print("error")
        imageIndex = imageIndex + 1
        continue
```
then by choosing 50 random bird images with hand and transfring them to ./data/test/birds, test and traning data preperation of birds are done.

## Data for frogs
64x64, RGB iamges of frogs are downloaded from https://images.cv/dataset/frog-image-classification-dataset, RGB frog images then turned into greyscale and saved as 450 traning and 50 test data to respected folders.
```
#RGB frog images is donwloaded from https://images.cv/dataset/frog-image-classification-dataset
for filename in os.listdir('./data/train/frogs'):
    f = os.path.join('./data/train/frogs', filename)
    image = cv2.imread(f,0)                             #reading greyscale frog image from training data for frogs (0->greyscale)
    image = cv2.resize(image,(64,64), interpolation = cv2.INTER_AREA) #resizeing greyscale frog image to 64x64
    cv2.imwrite('./data/train/frogs/{}.jpg'.format(filename), image)  #saving prepared image to training images of frogs

#choosing 50 random frog images with hand and transfering them to ./data/test/frogss
```
# Usin k-NN algorithm
## Steps
For any given test image, we need to turn 64x64 matrix of values from 0 to 255 into 4096 dimentional vetor using .ravel() function of numpy for calculating euler distance from test image to training image in proccess of calculation.
```
....
testImage = cv2.imread(testImageDir,0)         #loading 64x64 pixel greyscale image to 64x64 matrix with values from 0 to 255
testImageVec = testImage.ravel()    
....
frogTrainImage = cv2.imread(file,0)                 #load it to 64x64 matrix 
frogTrainImageVec = frogTrainImage.ravel()    
....
def evaluateNN(kNArray,trainImageVec,testImageVec,label,fileName):
    dist = np.linalg.norm(trainImageVec - testImageVec) #euler distance between test and training data
    return {
            "label":label,                              
            "fileName": fileName,
            "distance": dist
    }
```
sorting leastKNeighbours array which contains all euler distances of givin test data to all traing datas, results in neighbours in incresing distances.
Using majorty vote methode with first 'k' closest neigbours k-NN algoritm makes its decition.
```
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
```
## Calculating Accuracy
After calculating all guesses for all training data, we just need to compare guessed and real values and store right and wrong guesses. 
```
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
```
# Results
This operation is performed with diffrent k values (1,3,5,7,9) and diffrent epochs for both traning and test data for seeing the effects of diffrent paramters to resulting accuracies.
## Performance of k-NN algorithm for diffrent k values and epoch limits (Test Curve)

![alt text](https://github.com/ufukky/FIZ437E/blob/main/Homework-1%20kNN/matlab/testCurve_EpochvsAcc_k139.jpg?raw=true)

As aspected, with higher epoch (Training Sample), accuracy of k-NN algorithm is increased and apporoched to %64 percent.

## Traing Curve
![alt text](https://github.com/ufukky/FIZ437E/blob/main/Homework-1%20kNN/matlab/trainingCurve_AccsK_k13579_epochall.jpg?raw=true)

With k=1 its is guarentied to have a traning data with distace zero (which is itself), so training accucary of k-NN with k=1 is always %100.
But all k's > 1 will results traning data itself (100 percent right guess) and other neighbours which can be falsely callasfied, therefore with increaseing k value, we can expect to training accuracy to approach same accuracy as test curve (~ 0.64)
