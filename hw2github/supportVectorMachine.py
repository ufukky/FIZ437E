import numpy as np 

class SupportVectorMachine:
    def __init__(self):
        #lbd : lambda paramter
        self.epoch = None
        self.b = None
        self.w = None
        self.losses = None

    def max(self,y,y_):
        ret = []
        for n in range(len(y)):
            val = y[n] * y_[n] 
            if val >= 1:
                ret.append(0)
            else:
                ret.append(1-val)
        return np.array(ret)

    def loss(self,y,yt,lbd):
        dot = np.dot(self.w,self.w)
        m = self.max(y,yt)
        return lbd * np.dot(self.w,self.w) + np.mean(self.max(y,yt))

    def sign(self,x):
        if x <= 0:
            return -1
        else:
            return 1

    def predict(self,x):
        predictions = []
        m,n = x.shape
        for i in range(m):
            predictions.append(self.sign(np.dot(x[i],self.w) + self.b))
        return np.array(predictions)

    def train(self,x,y,lbd,lr=0.0001,bs=10,epochs=1000):
        # x --> Input.
        # y --> true/target value.
        # bs --> Batch Size.
        # epochs --> Number of iterations.
        # lr --> Learning rate.
            
        # m-> number of training examples
        # n-> number of features         
        m, n = x.shape        
        
        # Initializing weights and bias to zeros.
        self.w = np.zeros((n))
        self.b = 0

        # Empty list to store losses.
        self.losses = []

        # Training loop.
        for epoch in range(epochs):
            for i,xi, in enumerate(x):
                yi = y[i]
                if yi * self.sign(np.dot(xi,self.w) + self.b) >= 1:
                    self.w -= lr*lbd*2*self.w
                else:
                    self.w -= lr*(2*lbd*self.w - yi*xi)
                    self.b -= lr*yi
            # Calculating loss and appending it in the list.
            preds = self.predict(x)
            l = self.loss(y,preds,lbd)
            self.losses.append(l)




class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []

    def max(self,y,pred):
        if y * pred >= 1:
            return 0
        else :
            return 1

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            preds = self.predict(X)
            l = 0
            for i in range(len(preds)):
                
                l+= y[i]*self.max(y[i],preds[i]) + (1-y[i] * self.max(y[i],preds[i]))
            l += np.dot(self.w,self.w) * self.lambda_param
            self.losses.append(l)
        self.losses = np.array(self.losses)


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)