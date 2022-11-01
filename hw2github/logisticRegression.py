import numpy as np

class LogisticRegression:

    def __init__(self,epochs = 1000,learningRate = 0.001):
        self.epoch = epochs
        self.lr = learningRate
        self.b = None
        self.w = None
        self.losses = None
    
    def sigmoid(self,z):
        return 1.0/(1 + np.exp(-z))

    def loss(self,y, y_):
        return np.mean(y*(np.log(y_)) - (1-y)*np.log(1-y_))    
    
    def gradients(self,X, y, y_hat):
        # X --> Input.
        # y --> true/target value.
        # y_hat --> hypothesis/predictions.
        # w --> weights (parameter).
        # b --> bias (parameter).
        
        # m-> number of training examples.
        m = X.shape[0]
        
        # Gradient of loss w.r.t weights.
        dw = (1/m)*np.dot(X.T, (y_hat - y))
        
        # Gradient of loss w.r.t bias.
        db = (1/m)*np.sum((y_hat - y)) 
        
        return dw, db

    def train(self,xTrain, yTrain, batchSize, epochs, lr):
    
        # X --> Input.
        # y --> true/target value.
        # bs --> Batch Size.
        # epochs --> Number of iterations.
        # lr --> Learning rate.
            
        # m-> number of training examples
        # n-> number of features         
        m, n = xTrain.shape        
        
        # Initializing weights and bias to zeros.
        w = np.zeros((n,1))
        b = 0

        # Empty list to store losses.
        self.losses = []
        
        # Training loop.
        for epoch in range(epochs):
            for i in range((m-1)//batchSize + 1):
                
                # Defining batches. SGD.
                start = i*batchSize
                end = start + batchSize
                xb = xTrain[start:end]
                yb = yTrain[start:end]
                
                # Calculating hypothesis/prediction.
                y_ = self.sigmoid(np.dot(xb, w) + b)
                
                # Getting the gradients of loss w.r.t parameters.
                dw, db = self.gradients(xb, yb, y_)
                
                # Updating the parameters.
                w -= lr*dw
                b -= lr*db
            
            # Calculating loss and appending it in the list.
            l = self.loss(yTrain, self.sigmoid(np.dot(xTrain, w) + b))
            self.losses.append(l)
        self.w, self.b = w,b

    def predict(self,X):
        preds = self.sigmoid(np.dot(X,self.w) + self.b)
        predictions = np.array([[0] for i in range(len(preds))])
        for i in range(len(preds)):
            p = preds[i][0]
            if(p >= 0.5):
                predictions[i][0] = 1
            else:
                predictions[i][0] = 0
        return predictions

