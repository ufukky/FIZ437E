#Ufuk Köksal Yücedağ 090180117

import requests
import cv2
import numpy as np
import os

#----------------------------------------------------------frogs-----------------------------------------------------------
#RGB frog images is donwloaded from https://images.cv/dataset/frog-image-classification-dataset
for filename in os.listdir('./data/train/frogs'):
    f = os.path.join('./data/train/frogs', filename)
    image = cv2.imread(f,0)                             #reading greyscale frog image from training data for frogs (0->greyscale)
    image = cv2.resize(image,(64,64), interpolation = cv2.INTER_AREA) #resizeing greyscale frog image to 64x64
    cv2.imwrite('./data/train/frogs/{}.jpg'.format(filename), image)  #saving prepared image to training images of frogs

#choosing 50 random frog images with hand and transfering them to ./data/test/frogss

#---------------------------------------------------------birds---------------------------------------------------------------
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
#choosing 50 random bird images with hand and transfring them to ./data/test/birds

