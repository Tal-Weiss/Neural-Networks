import numpy as np

class Activation:
    
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        elif self.activation_type == 'none':
            return x
        else:
            raise ValueError("Unsupported activation type")

    def backward(self, x):
        if self.activation_type == 'tanh':
            return (1 - np.tanh(x) ** 2)
        elif self.activation_type == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_type == 'softmax':
            return x
        elif self.activation_type == 'none':
            return 1
        else:
            raise ValueError("Unsupported activation type")