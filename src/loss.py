import numpy as np

class Loss:
    def __init__(self):
        pass

    # Classifier - softmax
    def softmax(self, X, W, b):
        XTW = np.dot(X.T, W) + b.T  
        eta = np.max(XTW, axis=1, keepdims=True)
        e_XTW = np.exp(XTW - eta)
        softmax_probs = e_XTW / np.sum(e_XTW, axis=1, keepdims=True)
        return softmax_probs
    
    def softmax_predictions(self, probabilities):
        return np.argmax(probabilities, axis=1).reshape(-1, 1)
    
    def cross_entropy_loss(self, predictions, C):
        m = C.shape[0]
        # 1e-9 Adding a small value to avoid log(0)
        log_predictions = np.log(predictions + 1e-9)  
        loss = (-1/m) * np.sum(C * log_predictions)
        return loss
    
    def cross_entropy_gradient(self, predictions, C, X, W):
        m = X.shape[1]
        gradient = (predictions - C) / m
        grad_W = np.dot(X, gradient)
        grad_b = np.sum(gradient, axis=0, keepdims=True).T
        grad_X = np.dot(W, gradient.T)
        return grad_W, grad_b, grad_X
    
    
    # Least squares
    def least_squares_loss(self, X, y, W, b):
        predictions = X.T @ W + b.T 
        y.reshape(-1, 1)  
        errors = predictions - y
        cost = (1 / X.shape[1]) * np.sum(errors ** 2)
        return cost

    def least_squares_gradient(self, X, y, W, b):
        m = X.shape[1]  
        predictions = X.T @ W + b.T  
        errors = predictions - y  
        grad_W = (2 / m) * (X @ errors) 
        grad_b = (2 / m) * np.sum(errors, axis=0, keepdims=True).T 
        return grad_W, grad_b

    def least_squares_predictions(self, X, W, b):
        return X.T @ W + b.T


