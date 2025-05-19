import numpy as np
import matplotlib.pyplot as plt
from loss import Loss
from layer import NNLayer, ResNetLayer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# gradient test - softmax 
def grad_test(Data):
    # Extract data from the Data object
    X = Data.train_data
    C = Data.train_indicators
    W = Data.weights_train
    b = Data.bias_train
    data_name = Data.name

    # Initialize the loss object
    loss = Loss()

    # Compute the initial softmax loss and gradient
    prediction = loss.softmax(X, W, b)  
    F0 = loss.cross_entropy_loss(prediction, C)
    G0_W, G0_b, G0_X = loss.cross_entropy_gradient(prediction, C, X, W)

    # Initialize direction matrices
    D_W = np.random.rand(W.shape[0], W.shape[1])
    d_b = np.random.rand(b.shape[0], b.shape[1])
    d_X = np.random.rand(X.shape[0], X.shape[1])

    # Set initial epsilon value for the test
    epsilon = 0.1

    # Arrays for zero-order and first-order approximations W, b, and X
    zero_order_W = np.zeros(10)  
    first_order_W = np.zeros(10) 
    zero_order_b = np.zeros(10) 
    first_order_b = np.zeros(10)  
    zero_order_X = np.zeros(10)  
    first_order_X = np.zeros(10)  

    for k in range(10):
        # Reduce epsilon geometrically
        epsk = epsilon * (0.5 ** k)

        # Perturbed W, b, and X
        W_prime = W + epsk * D_W
        b_prime = b + epsk * d_b
        X_prime = X + epsk * d_X

        # Compute softmax loss for perturbed W, b, and X
        prediction_W = loss.softmax(X, W_prime, b)
        prediction_b = loss.softmax(X, W, b_prime)
        prediction_X = loss.softmax(X_prime, W, b)
        
        Fk_W = loss.cross_entropy_loss(prediction_W, C)
        Fk_b = loss.cross_entropy_loss(prediction_b, C)
        Fk_X = loss.cross_entropy_loss(prediction_X, C)

        # First-order approximation of Fk using gradients
        F1_W = F0 + epsk * np.sum(G0_W * D_W)
        F1_b = F0 + epsk * np.sum(G0_b * d_b)
        F1_X = F0 + epsk * np.sum(G0_X * d_X)

        # Absolute errors for zero-order and first-order
        zero_order_W[k] = abs(Fk_W - F0)
        first_order_W[k] = abs(Fk_W - F1_W)
        zero_order_b[k] = abs(Fk_b - F0)
        first_order_b[k] = abs(Fk_b - F1_b)
        zero_order_X[k] = abs(Fk_X - F0)
        first_order_X[k] = abs(Fk_X - F1_X)


    # Plot the errors in a semilogarithmic plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.semilogy(range(1, 11), zero_order_W, label="Zero order approx (W)")
    plt.semilogy(range(1, 11), first_order_W, label="First order approx (W)")
    plt.legend()
    plt.title(f"Gradient Test for Weights (W) - {data_name}")
    plt.xlabel("k")
    plt.xticks(range(1, 11))
    plt.ylabel("Error")

    plt.subplot(1, 3, 2)
    plt.semilogy(range(1, 11), zero_order_b, label="Zero order approx (b)")
    plt.semilogy(range(1, 11), first_order_b, label="First order approx (b)")
    plt.legend()
    plt.title(f"Gradient Test for Biases (b) - {data_name}")
    plt.xlabel("k")
    plt.xticks(range(1, 11))
    plt.ylabel("Error")

    plt.subplot(1, 3, 3)
    plt.semilogy(range(1, 11), zero_order_X, label="Zero order approx (X)")
    plt.semilogy(range(1, 11), first_order_X, label="First order approx (X)")
    plt.legend()
    plt.title(f"Gradient Test for Input Data (X) - {data_name}")
    plt.xlabel("k")
    plt.xticks(range(1, 11))
    plt.ylabel("Error")

    plt.suptitle(f"Gradient Test Results - {data_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
 
 
# Jacobian test - layers of the network    
def jacobian_test_network(model, X, sample_num=1, plot=True, parent_frame=None):
    
    model.forward(X)
    layers_num = model.num_layers
    
    # Loop over all layers - except the softmax layer
    for i in range(layers_num-1):
        curr_layer = model.layers[i]
        v = np.random.rand(curr_layer.output.shape[0], curr_layer.input.shape[1])
        v /= np.linalg.norm(v) if np.linalg.norm(v) != 0 else 1
        X = curr_layer.input.astype(np.float64)  # Avoid overflow
        base_forward = np.vdot(v, curr_layer.output)
                
        if isinstance(curr_layer, ResNetLayer): 
            grad_x, grad_w1, grad_w2, grad_b = curr_layer.backward(v)
            
            # Testing W1
            zero_order_w1, first_order_w1 = test_gradients(curr_layer.W1, grad_w1, sample_num=sample_num, 
                                                           X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing W2
            zero_order_w2, first_order_w2 = test_gradients(curr_layer.W2, grad_w2, sample_num=sample_num, 
                                                           X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing b
            zero_order_b, first_order_b = test_gradients(curr_layer.b, grad_b, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing X
            zero_order_x, first_order_x = test_gradients(curr_layer.input, grad_x, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            if plot:
                gradients = [
                    ("W1", zero_order_w1, first_order_w1),
                    ("W2", zero_order_w2, first_order_w2),
                    ("b", zero_order_b, first_order_b),
                    ("X", zero_order_x, first_order_x)
                ]
                plot_gradients(gradients, "ResNetLayer", i+1, parent_frame=parent_frame)
        else:   
            grad_x, grad_w, grad_b = curr_layer.backward(v)
             
            # Testing W
            zero_order_w, first_order_w = test_gradients(curr_layer.Weights, grad_w, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing b
            zero_order_b, first_order_b = test_gradients(curr_layer.Bias, grad_b, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing X
            zero_order_x, first_order_x = test_gradients(curr_layer.input, grad_x, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            
            if plot:
                gradients = [
                    ("W", zero_order_w, first_order_w),
                    ("b", zero_order_b, first_order_b),
                    ("X", zero_order_x, first_order_x)
                ]
                plot_gradients(gradients, "NNLayer", i+1, parent_frame=parent_frame)
            
def test_gradients(parameter, grad_param, sample_num, X, v, curr_layer, base_forward):
    
    epsilon_iterator = [0.5 ** i for i in range(1, 11)]
    
    # Initialize accumulators for differences
    zero_order = np.zeros(len(epsilon_iterator))
    first_order = np.zeros(len(epsilon_iterator))
                    
    for i in range(sample_num):
        # Generate a random perturbation
        perturbations = np.random.randn(*parameter.shape)
        perturbations /= np.linalg.norm(perturbations) if np.linalg.norm(perturbations) != 0 else 1
                        
        original_param = parameter.copy()
                        
        for idx, eps in enumerate(epsilon_iterator):
            
            # Perturb the parameter
            parameter += perturbations * eps
            # Forward pass after perturbation
            forward_after_eps = np.vdot(v, curr_layer.forward(X))
            # Revert the parameter to original
            parameter[:] = original_param
                            
            # Compute differences
            diff = np.abs(forward_after_eps - base_forward)
            grad_diff = np.abs(forward_after_eps - base_forward - np.vdot(grad_param, perturbations * eps))
                            
            # Accumulate differences
            zero_order[idx] += diff
            first_order[idx] += grad_diff
                    
    # Compute average over samples
    avg_zero_order = zero_order / sample_num
    avg_first_order = first_order / sample_num
                    
    return avg_zero_order, avg_first_order

def plot_gradients(gradients, layer_type, layer_num, parent_frame=None):
    epsilon_iterator = [0.5 ** i for i in range(1, 11)]
    x_labels = list(range(1, 11))  # Labels from 1 to 10
    
    num_plots = len(gradients)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    for idx, (param_name, avg_grad_diffs, avg_grad_diffs_grad) in enumerate(gradients):
        axs[idx].plot(x_labels, avg_grad_diffs, label=f"Zero-order approximation ({param_name})")
        axs[idx].plot(x_labels, avg_grad_diffs_grad, label=f"First-order approximation ({param_name})")
        axs[idx].set_yscale('log')
        axs[idx].set_ylabel('Error')
        axs[idx].set_xlabel('k')
        axs[idx].set_title(f'Error vs. k for {param_name}')
        axs[idx].legend()
        axs[idx].set_aspect('auto')
        axs[idx].set_xticks(x_labels)
    
    plt.suptitle(f'Gradient Test for {layer_type}, Layer: {layer_num}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)
    if parent_frame:
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill='both', expand=True)
    else:
        plt.show()


# Gradient test - entire network
def gradient_test_network(model, X, C, num_samples=1, plot=True, parent_frame=None):

    epsilon_iterator = [0.5 ** i for i in range(1, 11)]
    is_resNet = False

    zero_order_errors = np.zeros(len(epsilon_iterator))
    first_order_errors = np.zeros(len(epsilon_iterator))

    prediction = model.forward(X)
    F0 = Loss().cross_entropy_loss(prediction, C)
    grads = model.backprop(prediction, C)

    flat_grads = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, ResNetLayer):
            is_resNet = True
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_W1', np.zeros_like(layer.W1)).flatten())
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_W2', np.zeros_like(layer.W2)).flatten())
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_b', np.zeros_like(layer.b)).flatten())
        else:
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_W', np.zeros_like(layer.Weights)).flatten())
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_b', np.zeros_like(layer.Bias)).flatten())
    flat_grads = np.concatenate(flat_grads)

    original_params = model.get_params().copy()

    for idx in range(num_samples):
        perturbations = np.random.randn(len(flat_grads))
        perturbations /= np.linalg.norm(perturbations) if np.linalg.norm(perturbations) != 0 else 1.0

        for k, eps in enumerate(epsilon_iterator):
            flat_params_plus = original_params + eps * perturbations

            model.set_params(flat_params_plus)
            prediction_perturbation = model.forward(X)
            F_plus = Loss().cross_entropy_loss(prediction_perturbation, C)
            model.set_params(original_params)

            zero_order_error = np.abs(F_plus - F0)
            first_order_error = np.abs(F_plus - F0 - np.vdot(flat_grads, eps * perturbations))

            zero_order_errors[k] += zero_order_error / num_samples
            first_order_errors[k] += first_order_error / num_samples

    if plot and parent_frame is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, 11), zero_order_errors, label="Zero order error")
        ax.plot(range(1, 11), first_order_errors, label="First order error")
        ax.set_xticks(range(1, 11))
        ax.set_yscale('log')
        ax.set_xlabel("k")
        ax.set_ylabel("Error")
        ax.set_title("Gradient Test - Residual Network" if is_resNet else "Gradient Test - Neural Network")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

    return zero_order_errors, first_order_errors