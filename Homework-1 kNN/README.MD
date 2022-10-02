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
#choosing 50 random bird images with hand and transfring them to ./data/test/birds
```
