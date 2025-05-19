import numpy as np
from Activation import Activation
from loss import Loss

class NNLayer:
    def __init__(self, input_dim, output_dim, activation):
        
        # Layer is defined as: Activation(Wx + b)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # activation function
        self.activation = Activation(activation)
        self.activation_type = activation
        
        # initialize weights
        # W dimension: output_dim x input_dim
        self.Weights = np.random.randn(output_dim, input_dim)
        self.Bias = np.random.randn(output_dim, 1)
        
        # initialize gradients
        self.grad_Weights = None
        self.grad_Bias = None
        
        # initialize input and output of the layer
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        result = self.Weights.dot(input) + self.Bias
        self.output = self.activation.forward(result)
        return self.output
    
    def backward(self, grad_propagte):
        # Softmax layer
        if self.activation.activation_type == 'softmax':
            pred, C = grad_propagte
            X = self.input
            Weights = self.Weights
            
            grad = Loss().cross_entropy_gradient(pred, C, X, Weights.T)
            self.grad_Weights = grad[0].T
            self.grad_Bias = grad[1]
            grad_X = grad[2]  
    
        # Normal layer
        else:
            # sigma'(Wx+b)*v
            grad = self.activation.backward(self.Weights.dot(self.input) + self.Bias) * grad_propagte 
            
            # gradient according to Weights, Bias and X
            self.grad_Weights = grad.dot(self.input.T)
            self.grad_Bias = np.sum(grad, axis=1, keepdims=True)
            grad_X = self.Weights.T.dot(grad)
        
        return grad_X, self.grad_Weights, self.grad_Bias
    
    def update(self, learning_rate):
        self.Weights -= learning_rate * self.grad_Weights
        self.Bias -= learning_rate * self.grad_Bias
        
    def get_params(self):
        # Return the layer's parameters as a vector
        return np.concatenate([self.Weights.flatten(), self.Bias.flatten()])

    def set_params(self, flat_params):
        # Set the layer's parameters from a vector
        W_size = self.Weights.size
        self.Weights = flat_params[:W_size].reshape(self.Weights.shape)
        self.Bias = flat_params[W_size:].reshape(self.Bias.shape)
        


class ResNetLayer:
    def __init__(self, input_dim, output_dim, activation):
        
        # Layer is defined as:  x + W2*Activation(W1x + b)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # activation function
        self.activation = Activation(activation)
        self.activation_type = activation
        
        # Initialize weights
        # W1 dimension: output_dim x input_dim
        self.W1 = np.random.randn(output_dim, input_dim)
        # W2 dimension: input_dim x output_dim
        self.W2 = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(output_dim, 1)
        
        # Initialize gradients
        self.grad_W1 = None
        self.grad_W2 = None
        self.grad_b = None
        
        # Initialize input and output
        self.input = None
        self.output = None
        
        # Activation(W1x + b)
        self.middle = None

    def forward(self, X):
        self.input = X
        self.middle = self.activation.forward(np.dot(self.W1, X) + self.b)
        self.output = X + np.dot(self.W2, self.middle)
        return self.output

    def backward(self, grad_propagate):
        grad = self.activation.backward(np.dot(self.W1, self.input) + self.b) * np.dot(self.W2.T, grad_propagate)
        
        self.grad_W1 = np.dot(grad, self.input.T)
        self.grad_W2 = np.dot(grad_propagate, self.middle.T)
        self.grad_b = np.sum(grad, axis=1, keepdims=True)
        grad_X = grad_propagate + np.dot(self.W1.T, grad)
        
        return grad_X, self.grad_W1, self.grad_W2, self.grad_b

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.grad_W1
        self.W2 -= learning_rate * self.grad_W2
        self.b -= learning_rate * self.grad_b
        
    def get_params(self):
        # Return the layer's parameters as a vector
        return np.concatenate([self.W1.flatten(), self.W2.flatten(), self.b.flatten()])
        
    def set_params(self, flat_params):
        # Set the layer's parameters from a vector
        W1_size = self.W1.size
        W2_size = self.W2.size
        b_size = self.b.size
        self.W1 = flat_params[:W1_size].reshape(self.W1.shape)
        self.W2 = flat_params[W1_size:W1_size + W2_size].reshape(self.W2.shape)
        self.b = flat_params[W1_size + W2_size:].reshape(self.b.shape)
        
    