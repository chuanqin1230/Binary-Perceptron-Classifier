import numpy as np
import pandas as pd

''' Binary classifier'''
# Binary classifier with perceptron 
class BinaryPerceptron(object):
    def __init__(self, iteration = 50):
        self.iteration = iteration
    
    def fit(self, X, y):
        print('Fit the binary classifier with perceptron:')
        self.weight = np.zeros(X.shape[1])
        self.errors_list = []
        self.accuracy_list = []
        self.rate = 1
        for _ in range(self.iteration):
            errors = 0
            for xt, target in zip(X, y):
                if self.predict(xt) != target:
                    self.weight += self.rate * target * xt
                    errors += 1
            self.errors_list.append(errors)
            self.accuracy_list.append(1-errors/len(y))
            print('Iteration '+ str(_) + ', number of errors = ' +  str(errors))
        return self
    
    def predict(self, X):
        return np.where(np.dot(self.weight, X) >= 0, 1, -1) 

    
# Binary classifier with passive-aggressive
class BinaryPA(object):
    def __init__(self, iteration = 50):
        self.iteration = iteration
        
    def fit(self, X, y):
        print('Fit the binary classifier with passive-aggressive algorithm:')
        self.weight = np.zeros(X.shape[1])
        self.errors_list = []
        self.accuracy_list = []
        self.rate = 0
        for _ in range(self.iteration):
            errors = 0
            for xt, target in zip(X, y):
                if self.predict(xt) != target:
                    self.rate = (1 - max(0,target * np.dot(self.weight, xt)))/(np.linalg.norm(xt))
                    self.weight += self.rate * target * xt
                    errors += 1
            self.errors_list.append(errors)
            self.accuracy_list.append(1-errors/len(y))
            print('Iteration '+ str(_) + ', number of errors = ' +  str(errors))
        return self
    
    def predict(self, X):
        return np.where(np.dot(self.weight, X) >= 0, 1, -1)


# Binary classifier with averaged perceptron 
class BinaryAveragedPerceptron(object):
    def __init__(self, iteration = 50):
        self.iteration = iteration
    
    def fit(self, X, y):
        print('Fit the binary classifier with perceptron:')
        self.weight = np.zeros(X.shape[1])
        self.averaged_weight = np.zeros(X.shape[1])
        self.errors_list = []
        self.accuracy_list = []
        self.rate = 1
        for _ in range(self.iteration):
            errors = 0
            for xt, target in zip(X, y):
                if self.predict(xt) != target:
                    self.weight += self.rate * target * xt
                    self.averaged_weight += self.weight
                    errors += 1
            self.errors_list.append(errors)
            self.accuracy_list.append(1-errors/len(y))
            print('Iteration '+ str(_) + ', number of errors = ' +  str(errors))
        return self
    
    def predict(self, X):
        return np.where(np.dot(self.averaged_weight, X) >= 0, 1, -1) 
		

''' Online Multi-Class Classifier Learning Algorithm'''
# Multi-class classifier with perceptron 
class MCPerceptron(object):
    def __init__(self, classes = 10, iteration = 50):
        self.classes = classes
        self.iteration = iteration
    
    def fit(self, X, y):
        print('Fit the multi-class classifier with perceptron:')
        self.weight = np.zeros([self.classes, X.shape[1]])
        self.errors_list = []
        self.accuracy_list = []
        self.rate = 1
        for _ in range(self.iteration):
            errors = 0
            for xt, target in zip(X, y):
                y_hat = self.predict(xt)
                if y_hat != target:
                    self.weight += np.dot(self.rate, self.augmented_features(xt, target) - self.augmented_features(xt, y_hat))
                    errors += 1
            self.errors_list.append(errors)
            self.accuracy_list.append(1-errors/len(y))
            print('Iteration '+ str(_) + ', number of errors = ' +  str(errors))
        return self
    
    def predict(self, X):
        prediction = (self.weight * self.augmented_features(X, range(self.classes))).sum(axis = 1)
        y_hat = np.argmax(prediction)
        return y_hat
    
    def augmented_features(self, X, y): 
        F = np.zeros((self.classes,len(X)))
        F[y] = X
        return F


# Multi-class classifier with Passive-Aggressive (PA) 
class MCPA(object):
    def __init__(self, classes = 10, iteration = 50):
        self.classes = classes
        self.iteration = iteration
    
    def fit(self, X, y):
        print('Fit the multi-class classifier with perceptron:')
        self.weight = np.zeros([self.classes, X.shape[1]])
        self.errors_list = []
        self.accuracy_list = []
        self.rate = 0.0
        for _ in range(self.iteration):
            errors = 0
            for xt, target in zip(X, y):
                y_hat = self.predict(xt)
                if y_hat != target:
                    loss = np.sum(np.dot(self.weight, self.augmented_features(xt, target)[target]) - np.dot(self.weight, self.augmented_features(xt, y_hat)[y_hat]))
                    norm2 = np.linalg.norm(self.augmented_features(xt, target) - self.augmented_features(xt, y_hat))
                    self.rate = (1 - max(0, loss))/norm2
#                     self.rate = 1
                    self.weight += np.dot(self.rate, self.augmented_features(xt, target) - self.augmented_features(xt, y_hat))
                    errors += 1
            self.errors_list.append(errors)
            self.accuracy_list.append(1-errors/len(y))
            print('Iteration '+ str(_) + ', number of errors = ' +  str(errors))
        return self
    
    def predict(self, X):
        prediction = (self.weight * self.augmented_features(X, range(self.classes))).sum(axis = 1)
        y_hat = np.argmax(prediction)
        return y_hat
    
    def augmented_features(self, X, y): 
        F = np.zeros((self.classes,len(X)))
        F[y] = X
        return F
		
''' Online Multi-Class Classifier Learning Algorithm'''
# Multi-class classifier with perceptron 
class MCAveragedPerceptron(object):
    def __init__(self, classes = 10, iteration = 50):
        self.classes = classes
        self.iteration = iteration
    
    def fit(self, X, y):
        print('Fit the multi-class classifier with perceptron:')
        self.weight = np.zeros([self.classes, X.shape[1]])
        self.averaged_weight = np.zeros([self.classes, X.shape[1]])
        self.errors_list = []
        self.accuracy_list = []
        self.rate = 1
        for _ in range(self.iteration):
            errors = 0
            for xt, target in zip(X, y):
                y_hat = self.predict(xt)
                if y_hat != target:
                    self.weight += np.dot(self.rate, self.augmented_features(xt, target) - self.augmented_features(xt, y_hat))
                    self.averaged_weight += self.weight
                    errors += 1
            self.errors_list.append(errors)
            self.accuracy_list.append(1-errors/len(y))
            print('Iteration '+ str(_) + ', number of errors = ' +  str(errors))
        return self
    
    def predict(self, X):
        prediction = (self.averaged_weight * self.augmented_features(X, range(self.classes))).sum(axis = 1)
        y_hat = np.argmax(prediction)
        return y_hat
    
    def augmented_features(self, X, y): 
        F = np.zeros((self.classes,len(X)))
        F[y] = X
        return F