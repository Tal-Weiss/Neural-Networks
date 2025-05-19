import numpy as np
from layer import ResNetLayer, NNLayer
from loss import Loss
from optimizer import Optimizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ResNet:
    def __init__(self, layers_config, learning_rate, batch_size):
        
        self.num_layers = len(layers_config)
        self.layers = []
        self.lr = learning_rate
        self.optimizer = Optimizer(learning_rate=learning_rate, batch_size=batch_size)
        
        for i, (input_dim, output_dim, activation) in enumerate(layers_config):
            if i == self.num_layers - 1:
                self.layers.append(NNLayer(input_dim, output_dim, activation))
            else:
                self.layers.append(ResNetLayer(input_dim, output_dim, activation))

    def print_model(self):
        print("ResNet Architecture:")
        for i, layer in enumerate(self.layers):
            layer_type = "ResNetLayer" if isinstance(layer, ResNetLayer) else "NNLayer"
            print(f" Layer {i+1} ({layer_type}): {layer.input_dim} → {layer.output_dim}  |  Activation: {layer.activation_type}")
    
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output.T

    def backward(self, pred, C):
        grad_propagate = pred, C    

        # Backward pass - all layers
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            grad_propagate = layer.backward(grad_propagate)[0]
            
        # Update parameters - all layers
        for layer in self.layers:
            layer.update(self.lr)
        
    
    # only for testing - backward with no update
    def backprop(self, pred, C):
       
        # Dictionary to store gradients for tests
        gradients = {}
        grad_propagate = pred, C    

        # Backward pass - all layers
        for i in reversed(range(self.num_layers)):
            
            layer = self.layers[i]
            grad_propagate = layer.backward(grad_propagate)[0]
            
            if isinstance(layer, NNLayer):
                # Store gradients for the layer
                gradients[f'layer_{i}'] = {
                    'grad_W': layer.grad_Weights,
                    'grad_b': layer.grad_Bias
                }
            else:
                # Store gradients for the layer
                gradients[f'layer_{i}'] = {
                    'grad_W1': layer.grad_W1,
                    'grad_W2': layer.grad_W2,
                    'grad_b': layer.grad_b
                }
    
        return gradients

    def train(self, X_train, y_train, C_train, epochs, batch_size, X_val=None, y_val=None, C_val=None):
        """
        Train the neural network.

        Parameters:
        X_train (numpy.ndarray): Training input data, shape (n, m) where n is the number of features and m is the number of samples.
        y_train (numpy.ndarray): Training true labels, shape (m,) where m is the number of samples.
        C_train (numpy.ndarray): Training indicators, shape (m, l) where m is the number of samples and l is the number of classes.
        epochs (int): Number of training epochs
        batch_size (int): Size of each mini-batch
        X_val (numpy.ndarray, optional): Validation input data, shape (n, m_val) where n is the number of features and m_val is the number of validation samples.
        y_val (numpy.ndarray, optional): Validation true labels, shape (m_val,) where m_val is the number of validation samples.
        C_val (numpy.ndarray, optional): Validation indicators, shape (m_val, l) where m_val is the number of validation samples and l is the number of classes.
        """
        loss_function = Loss()
        m = X_train.shape[1]

        metrics = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }

        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.arange(m)
            np.random.shuffle(indices)
            mini_batches = self.optimizer.create_mini_batches(indices, batch_size)

            epoch_loss = 0

            # Mini-batch training
            for mini_batch in mini_batches:
                X_batch = X_train[:, mini_batch]
                y_batch = y_train[mini_batch]
                C_batch = C_train[mini_batch, :]

                # Forward pass
                predictions = self.forward(X_batch)

                # Compute loss
                loss = loss_function.cross_entropy_loss(predictions, C_batch)
                epoch_loss += loss

                # Backward pass
                self.backward(predictions, C_batch)

            # Average loss for the epoch
            epoch_loss /= len(mini_batches)

            # Compute accuracy for the entire training set
            train_predictions = self.forward(X_train)
            train_predicted_classes = loss_function.softmax_predictions(train_predictions)
            train_accuracy = np.mean(train_predicted_classes == y_train)

            metrics['train_losses'].append(epoch_loss)
            metrics['train_accuracies'].append(train_accuracy)

            # Validation loss and accuracy
            if X_val is not None and y_val is not None and C_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = loss_function.cross_entropy_loss(val_predictions, C_val)
                val_predicted_classes = loss_function.softmax_predictions(val_predictions)
                val_accuracy = np.mean(val_predicted_classes == y_val)

                metrics['val_losses'].append(val_loss)
                metrics['val_accuracies'].append(val_accuracy)

                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss}, Training Accuracy: {train_accuracy}")

        return metrics
    
    def plot_metrics(self, metrics, parent=None):
        epochs = range(1, len(metrics['train_losses']) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot training and validation loss
        axs[0].plot(epochs, metrics['train_losses'], label='Training Loss')
        if metrics['val_losses']:
            axs[0].plot(epochs, metrics['val_losses'], label='Validation Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Plot training and validation accuracy
        axs[1].plot(epochs, metrics['train_accuracies'], label='Training Accuracy')
        if metrics['val_accuracies']:
            axs[1].plot(epochs, metrics['val_accuracies'], label='Validation Accuracy')
        axs[1].set_title('Training and Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.tight_layout()

        if parent is not None:
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(padx=10, pady=10)
            plt.close(fig)
        else:
            plt.show()
    
    
    
    def get_params(self):
        # Return all layers parameters as a single vector
        return np.concatenate([layer.get_params() for layer in self.layers])

    def set_params(self, flat_params):
        # Set all layer parameters from a single vector
        current_idx = 0
        for layer in self.layers:
            layer_params = layer.get_params()
            params_size = layer_params.size
            layer.set_params(flat_params[current_idx:current_idx + params_size])
            current_idx += params_size
    
    