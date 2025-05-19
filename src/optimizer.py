import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from loss import Loss

class Optimizer:
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def create_mini_batches(self, indices, batch_size):
        """
        Create mini-batches from a list of indices.

        Parameters:
        indices (numpy.ndarray or list): Array or list of data indices to be split into mini-batches.
        batch_size (int): The number of indices in each mini-batch.

        Returns:
        list: A list of numpy arrays, where each array contains indices for one mini-batch.
        """
        mini_batches = [
            indices[i:i + batch_size] for i in range(0, len(indices), batch_size)
        ]
        return mini_batches
    
    def SGD(self, X, y, C, W, b, loss, epochs, 
            X_val=None, y_val=None, C_val=None, plot=True, convergence_threshold=1e-6):
        
        m = X.shape[1]  
        indices = np.arange(m)  
        losses = [] 
        success_percentages = []  
        residual_norms = []  
        val_losses = []  
        val_success_percentages = []  

        for k in range(epochs):
            
            np.random.shuffle(indices)  # Shuffle data indices
            mini_batches = self.create_mini_batches(indices, self.batch_size)

            for mini_batch in mini_batches:
                # Extract mini-batch data
                X_batch = X[:, mini_batch]
                y_batch = y[mini_batch]
                
                # Compute gradient
                if len(C) > 0:
                    predictions = loss.softmax(X_batch, W, b)
                    C_batch = C[mini_batch, :]  
                    grad_W, grad_b, _ = loss.cross_entropy_gradient(predictions, C_batch, X_batch, W)
                else:
                    # just for least squares example
                    predictions = loss.least_squares_predictions(X_batch, W, b)
                    grad_W, grad_b = loss.least_squares_gradient(X_batch, y_batch, W, b)

                # Update weights and biases
                W -= self.learning_rate * grad_W
                b -= self.learning_rate * grad_b

            # Compute training loss
            if len(C) > 0:
                predictions = loss.softmax(X, W, b)
                running_loss = loss.cross_entropy_loss(predictions, C)
                predictions = loss.softmax_predictions(predictions)
            else:
                predictions = loss.least_squares_predictions(X, W, b)
                running_loss = loss.least_squares_loss(X, y, W, b)

            # Compute training success percentage
            success_count = np.sum(predictions == y)
            success_percentage = (success_count / m) * 100
            
            # Compute residual norm
            residual = predictions - y
            residual_norm = np.linalg.norm(residual)
            
            # Track training metrics
            losses.append(running_loss)
            success_percentages.append(success_percentage)
            residual_norms.append(residual_norm)
            
            # Compute validation loss and success percentage
            if X_val is not None and y_val is not None:
                val_predictions = loss.softmax(X_val, W, b)
                val_running_loss = loss.cross_entropy_loss(val_predictions, C_val)
                val_predictions = loss.softmax_predictions(val_predictions)
                val_success_count = np.sum(val_predictions == y_val)
                val_success_percentage = (val_success_count / X_val.shape[1]) * 100
                val_losses.append(val_running_loss)
                val_success_percentages.append(val_success_percentage)
        
            # Check for convergence
            if k > 0 and residual_norms[-1] / residual_norms[-2] < convergence_threshold:
                print(f"Convergence reached at iteration {k}")
                break

        if plot:
            self.plot_sgd_results(losses, success_percentages, "Training Loss and Success Percentage")
            if X_val is not None and y_val is not None:
                self.plot_sgd_results(val_losses, val_success_percentages, "Validation Loss and Success Percentage")

        return W, b, losses, success_percentages, val_losses, val_success_percentages


    def plot_sgd_results(self, losses, success_percentages, title, parent_frame=None):

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot loss values
        ax[0].plot(losses, label='Average F(W) per iteration', color='blue')
        ax[0].scatter(len(losses) - 1, losses[-1], color='red', marker='*', zorder=5,
                    label=f'Final F(W): {losses[-1]:.3f}')  # Mark the final value
        ax[0].set_xlabel('Full Iterations')
        ax[0].set_ylabel('Value of F(W)')
        ax[0].set_title('Evolution of F(W) during SGD (per full iteration)')
        ax[0].legend()

        # Plot success percentages
        ax[1].plot(success_percentages, label='Success Percentage', color='green')
        ax[1].scatter(len(success_percentages) - 1, success_percentages[-1], color='red', marker='*', zorder=5,
                    label=f'Final Success: {success_percentages[-1]:.3f}')  # Mark the final value
        ax[1].set_xlabel('Full Iterations')
        ax[1].set_ylabel('Success Percentage')
        ax[1].set_title('Success Percentage during SGD (per full iteration)')
        ax[1].legend()

        # Set the super title for the figure
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if parent_frame:
            # Embed the plot into the Tkinter frame
            canvas_plot = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(pady=10)
        else:
            # If no parent frame is provided, display in a new window
            plt.show()
        