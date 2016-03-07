import numpy as np

class Regression:
    '''
    Fit a linear regression model to the given data
    '''
    def __init__(self):
        '''
        Initialization for default parameters:
        max_iter=10000, lr=0.1, l2_penalty=0, degree=1, tolerance=1e-4 
        '''
        self.max_iter=10000
        self.lr = 0.1
        self.l2_penalty = 0
        self.degree=1
        self.theta=np.zeros((2,1))
        self.tolerance=1e-4
    
    def set_max_iter(self,iterations):
        '''
        set max iterations for the optimization
        @param iterations: max_iter
        '''
        self.max_iter = iterations
        
    def set_lr(self,learning_rate):
        '''
        set learning rate for the optimization
        @param learning rate: lr
        '''
        self.lr = learning_rate
        
    def set_l2_penalty(self, l2_penalty):
        '''
        set l2 regularizor for the optimization
        @param l2_penalty: l2_penalty
        '''
        self.l2_penalty = l2_penalty
        
    def set_tolerance(self,tolerance):
        '''
        set tolerance for the optimization
        @param tolerance: tolerance
        '''
        self.tolerance = tolerance
        
    def polynomial_fit(self, x, y, deg):
        '''
        curve fitting for the given data
        @param x: input vector
        @param y: target vector
        @param deg: ploynimial degree
        
        @return theta: model parameters
        @return loss: final calue of J(x,y,theta)
        @return repeat: final iterations of the optimization process
        
        This function evaluates: 
            minimize J(x,y,theta) = 1/(2*n)(sum(theta*x-y)^2 + sum(l2*theta^2))
            using gradient descent optimization method
        '''
        
        #feature matrix: [[x[0]**0,x[0]**1,...,x[0]**degree], [x[1]**0,x[1]**1,...,x[1]**degree], ... , [x[n-1]**0,x[n-1]**1,...,x[n-1]**degree]]
        self.degree = deg
        feature = np.zeros((len(x),deg+1))
        for i in np.arange(0,len(x)):
            for j in np.arange(0,deg):
                feature[i][j] = x[i]**j
        self.theta = np.zeros((deg+1,1))
        loss_old = loss = 0
        for repeat in np.arange(0,self.max_iter):
            error = np.dot(feature,self.theta)-y
            loss = np.dot(error.T,error)
            if(abs(loss-loss_old)<=self.tolerance):
                return self.theta, loss, repeat
            grad = np.dot(1.0/len(x), np.dot(feature.T, error))
            grad[1:,] += np.dot(self.l2_penalty,self.theta[1:,])
            self.theta -= np.dot(self.lr,grad)
            loss_old = loss
        return self.theta, loss, repeat
    
    def predict(self,x):
        '''
        Predict putput for unknown input data
        @param x: input data
        
        @return y: prediction output
        '''
        y = np.zeros((len(x),1))
        for i in np.arange(0,len(x)):
            for j in np.arange(0,self.degree):
                y[i] += self.theta[j]*(x[i]**j)
        return y