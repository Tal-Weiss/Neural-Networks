# FILE: /my-python-project/my-python-project/src/main.py
import matplotlib.pyplot as plt
from loss import Loss
from optimizer import Optimizer
import numpy as np
from Data import Data
from NeuralNetwork import NeuralNetwork
from ResNet import ResNet
from tests import jacobian_test_network, gradient_test_network, grad_test

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import threading

# Import the datasets
Swiss_Roll = Data("data/SwissRollData.mat", "Swiss Roll")
Peaks = Data("data/PeaksData.mat", "Peaks")
GMM = Data("data/GMMData.mat", "GMM")
datasets = {
    "Swiss Roll": Swiss_Roll,
    "Peaks": Peaks,
    "GMM": GMM
}

def default_neural_network(dataset):
    data = datasets[dataset]
    input = data.train_data.shape[0]
    output = data.train_indicators.shape[1]
    model = NeuralNetwork(
       layers_config=[(input, 10, 'tanh'),(10, 10, 'tanh'), (10, output, 'softmax')],
       learning_rate=0.01,
       batch_size=32
    )
    return model
   
def default_resnet(dataset):
    data = datasets[dataset]
    input = data.train_data.shape[0]
    output = data.train_indicators.shape[1]
    model = ResNet(
       layers_config=[(input, 10, 'tanh'),(input, 10, 'tanh'), (input, output, 'softmax')],
       learning_rate=0.01,
       batch_size=32
    )
    return model

def costumized_network(dataset, type, layers, lr, batch):
    data = datasets[dataset]
    input_dim = data.train_data.shape[0]
    output_dim = data.train_indicators.shape[1]
    
    # layers is a list of tuples: [(neurons, activation), ...]
    layer_config = []
    if type == "NeuralNetwork":
        for i, (neurons, activation) in enumerate(layers):
            if i == 0:
                # First layer connects to input_dim
                layer_config.append((input_dim, neurons, activation))
            else:
                # Subsequent layers connect to the previous layer's neurons
                layer_config.append((layers[i-1][0], neurons, activation))
        # Add the output layer
        layer_config.append((layers[-1][0], output_dim, 'softmax'))
        
        model = NeuralNetwork(
            layers_config=layer_config,
            learning_rate=lr,
            batch_size=batch
        )
    elif type == "ResNet":
        for i, (neurons, activation) in enumerate(layers):
            if i == 0:
                # First layer connects to input_dim
                layer_config.append((input_dim, neurons, activation))
            else:
                # Subsequent layers connect to the previous layer's neurons
                layer_config.append((layers[i-1][0], neurons, activation))
        # Add the output layer
        layer_config.append((layers[-1][0], output_dim, 'softmax'))
        
        model = ResNet(
            layers_config=layer_config,
            learning_rate=lr,
            batch_size=batch
        )
    else:
        raise ValueError("Invalid model type. Choose 'NeuralNetwork' or 'ResNet'.")
    return model

# Initialize Tkinter root FIRST
root = tk.Tk()
root.title("Neural Network GUI")
root.geometry("600x600")

def clear_window():
    # # Reset stdout and stderr to their original values
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__

    # Destroy all widgets in the root window
    for widget in root.winfo_children():
        widget.destroy()
        
def get_button_width(text):
    # This calculates the width dynamically based on the text length
    return max(30, len(text) + 2)

def show_start_menu():
    clear_window()
    tk.Label(root, text="Welcome to the Neural Network GUI", font=("Helvetica", 18, "bold")).pack(pady=20)
    
    # Add an image to the start menu (make sure you have an image file in your directory)
    try:
        img = Image.open("images/deep-neural-network_orig.png")
        img = img.resize((400, 300),Image.Resampling.LANCZOS)  # Resize the image
        img_tk = ImageTk.PhotoImage(img)
        image_label = tk.Label(root, image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
        image_label.pack(pady=20)
    except FileNotFoundError:
        print("Image file not found. Please make sure 'deep-neural-network_orig.png' is in your project directory.")
    
    # Start button
    tk.Button(root, text="Start", width=get_button_width("Start"), command=show_main_menu).pack(pady=10)
    tk.Button(root, text="Exit", width=get_button_width("Exit"), command=root.quit).pack(pady=10)

def show_main_menu():
    clear_window()
    tk.Label(root, text="Choose a dataset:", font=("Helvetica", 14, "bold")).pack(pady=5)
    for dataset in datasets.keys():
        tk.Button(root, text=dataset, width=get_button_width(dataset), command=lambda d=dataset: show_operation_menu(d)).pack(pady=5)
    tk.Button(root, text="Back", width=get_button_width("Back"), command=show_start_menu).pack(pady=10)

def show_operation_menu(dataset):
    clear_window()
    tk.Label(root, text=f"Selected Dataset: {dataset}", font=("Helvetica", 12, "italic")).pack(pady=5)
    tk.Label(root, text="Choose an operation:", font=("Helvetica", 14, "bold")).pack(pady=5)
    tk.Button(root, text="Find Best Hyperparameters", width=get_button_width("Find Best Hyperparameters"), command=lambda: find_best_hyperparams(dataset)).pack(pady=5)
    tk.Button(root, text="Train", width=get_button_width("Train"), command=lambda: show_train_menu(dataset)).pack(pady=5)
    tk.Button(root, text="Run Experiment", width=get_button_width("Run Experiment"), command=lambda: show_experiment_menu(dataset)).pack(pady=5)
    tk.Button(root, text="Tests", width=get_button_width("Tests"), command=lambda: show_tests_menu(dataset)).pack(pady=5)
    tk.Button(root, text="Back", width=get_button_width("Back"), command=show_main_menu).pack(pady=10)

def show_train_menu(dataset):
    clear_window()
    tk.Label(root, text=f"Training on {dataset}", font=("Helvetica", 12, "italic")).pack(pady=5)
    tk.Label(root, text="Choose Architecture:", font=("Helvetica", 14, "bold")).pack(pady=5)
    tk.Button(root, text="NeuralNetwork", width=get_button_width("NeuralNetwork"), command=lambda: show_train_options(dataset, "NeuralNetwork")).pack(pady=5)
    tk.Button(root, text="ResNet", width=get_button_width("ResNet"), command=lambda: show_train_options(dataset, "ResNet")).pack(pady=5)
    tk.Button(root, text="Back", width=get_button_width("Back"), command=lambda: show_operation_menu(dataset)).pack(pady=10)

def show_train_options(dataset, type):
    clear_window()
    tk.Label(root, text=f"{type} Training Options on {dataset}", font=("Helvetica", 12, "italic")).pack(pady=5)
    tk.Button(root, text="Default", width=get_button_width("Default"), command=lambda: train_model(dataset, type, default=True)).pack(pady=5)
    tk.Button(root, text="Customize", width=get_button_width("Customize"), command=lambda: show_custom_train_menu(dataset, type)).pack(pady=5)
    tk.Button(root, text="Back", width=get_button_width("Back"), command=lambda: show_train_menu(dataset)).pack(pady=10)

def show_custom_train_menu(dataset, model_type):
    clear_window()
    root.grid_rowconfigure(0, weight=0)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=0)
    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)

    tk.Label(root, text=f"Customize {model_type} Training on {dataset}", font=("Helvetica", 14, "bold")).grid(row=0, column=0, pady=10)

    # Frame for number of layers input and OK button
    layers_frame = tk.Frame(root)
    layers_frame.grid(row=1, column=0, pady=(0, 2))

    tk.Label(layers_frame, text="Number of Hidden Layers:", font=("Helvetica", 12)).pack(side='left', padx=5)

    def validate_numeric_input(P):
        return P.isdigit() or P == ""

    validate_command = root.register(validate_numeric_input)

    layers_entry = tk.Entry(layers_frame, width=5, validate="key", validatecommand=(validate_command, "%P"))
    layers_entry.pack(side='left', padx=5)

    tk.Button(layers_frame, text="OK", width=10, command=lambda: generate_layer_inputs()).pack(side='left', padx=5)

    # Sticky header row
    header_frame = tk.Frame(root)
    header_frame.grid(row=2, column=0, sticky="n", pady=(0, 2))
    tk.Label(header_frame, text="Layer", font=("Helvetica", 12, "bold"), width=10, anchor='center').grid(row=0, column=0, padx=5)
    tk.Label(header_frame, text="Neurons", font=("Helvetica", 12, "bold"), width=10, anchor='center').grid(row=0, column=1, padx=5)
    tk.Label(header_frame, text="Activation", font=("Helvetica", 12, "bold"), width=12, anchor='center').grid(row=0, column=2, padx=5)

    # Scrollable area setup
    canvas_frame = tk.Frame(root)
    canvas_frame.grid(row=3, column=0, pady=(2, 10), sticky="nsew")
    root.grid_rowconfigure(3, weight=1)

    canvas = tk.Canvas(canvas_frame, width=500, highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    scrollable_frame = tk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def on_configure(event):
        canvas_width = event.width
        canvas.itemconfig(window_id, width=canvas_width)
        if scrollable_frame.winfo_height() <= canvas.winfo_height():
            canvas.configure(yscrollcommand=None)
            scrollbar.pack_forget()
        else:
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")

    canvas.bind("<Configure>", on_configure)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    layer_entries = []
    activation_choices = []

    def generate_layer_inputs():
        nonlocal layer_entries, activation_choices
        layer_entries.clear()
        activation_choices.clear()

        for widget in scrollable_frame.winfo_children():
            widget.destroy()

        try:
            num_layers = int(layers_entry.get().strip())
            if num_layers <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid positive number of layers")
            return

        for i in range(num_layers):
            row_frame = tk.Frame(scrollable_frame)
            row_frame.grid(row=i, column=0, pady=2, padx=20)

            row_frame.grid_columnconfigure(0, weight=1)
            row_frame.grid_columnconfigure(1, weight=1)
            row_frame.grid_columnconfigure(2, weight=1)

            tk.Label(row_frame, text=f"Layer {i+1}:", font=("Helvetica", 12), width=10, anchor='center').grid(row=0, column=0, padx=5)

            neurons_entry = tk.Entry(row_frame, width=10, justify='center', validate="key", validatecommand=(validate_command, "%P"))
            neurons_entry.grid(row=0, column=1, padx=5)
            layer_entries.append(neurons_entry)

            activation = ttk.Combobox(row_frame, values=["relu", "tanh", "none"], width=12, justify='center', state='readonly')
            activation.grid(row=0, column=2, padx=5)
            activation.set("relu")  # <-- Add this line
            activation_choices.append(activation)

        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)

    # Frame for learning rate input
    learning_rate_frame = tk.Frame(root)
    learning_rate_frame.grid(row=4, column=0, pady=(5, 2))

    def validate_float_input(P):
        return P == "" or P.replace(".", "", 1).isdigit()

    validate_float_command = root.register(validate_float_input)

    tk.Label(learning_rate_frame, text="Learning Rate:", font=("Helvetica", 12)).pack(side='left', padx=5)
    learning_rate_entry = tk.Entry(learning_rate_frame, width=10, validate="key", validatecommand=(validate_float_command, "%P"))
    learning_rate_entry.pack(side='left', padx=5)

    # Frame for batch size input
    batch_size_frame = tk.Frame(root)
    batch_size_frame.grid(row=5, column=0, pady=(5, 2))

    tk.Label(batch_size_frame, text="Batch Size:", font=("Helvetica", 12)).pack(side='left', padx=5)
    batch_size_entry = tk.Entry(batch_size_frame, width=10, validate="key", validatecommand=(validate_command, "%P"))
    batch_size_entry.pack(side='left', padx=5)

    def start_training():
        neurons_per_layer = []
        activations = []

        # Validate layer configuration fields
        for idx, (entry, activation) in enumerate(zip(layer_entries, activation_choices)):
            n_val = entry.get().strip()
            a_val = activation.get().strip()

            if n_val == "" or a_val == "":
                messagebox.showwarning("Incomplete Input", f"Please fill in all fields for Layer {idx + 1}.")
                return

            try:
                neurons_per_layer.append(int(n_val))
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid number of neurons for Layer {idx + 1}. Must be an integer.")
                return

            activations.append(a_val)

        # Validate learning rate
        try:
            learning_rate = float(learning_rate_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid float for Learning Rate.")
            return

        # Validate batch size
        try:
            batch_size = int(batch_size_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid integer for Batch Size.")
            return

        layers = list(zip(neurons_per_layer, activations))

        train_model(dataset, model_type, layers=layers, lr=learning_rate, batch_size=batch_size, default=False)

    tk.Button(root, text="Start Training", width=20, command=start_training).grid(row=6, column=0, pady=5)
    tk.Button(root, text="Back", width=20, command=lambda: show_train_options(dataset, model_type)).grid(row=7, column=0, pady=10)

def show_experiment_menu(dataset):
    clear_window()
    tk.Label(root, text=f"Experiments on {dataset}", font=("Helvetica", 12, "italic")).pack(pady=5)
    tk.Button(root, text="Depth Experiment NN", width=get_button_width("Depth Experiment NN"), command=lambda: run_experiment(dataset, 1)).pack(pady=5)
    tk.Button(root, text="Depth Experiment ResNet", width=get_button_width("Depth Experiment ResNet"), command=lambda: run_experiment(dataset, 2)).pack(pady=5)
    tk.Button(root, text="Parameter Restiction NN", width=get_button_width("Parameter Restiction NN"), command=lambda: run_experiment(dataset, 3)).pack(pady=5)
    tk.Button(root, text="Back", width=get_button_width("Back"), command=lambda: show_operation_menu(dataset)).pack(pady=10)

def show_tests_menu(dataset):
    clear_window()
    tk.Label(root, text=f"Tests on {dataset} (defualt architecture)", font=("Helvetica", 12, "italic")).pack(pady=5)
    tk.Button(root, text="Gradient Test",  width=get_button_width("Gradient Test"), command=lambda: run_gradient_test_network(dataset)).pack(pady=5)
    tk.Button(root, text="Jacobian Test", width=get_button_width("Jacobian Test"), command=lambda: run_jacobian_test_network(dataset)).pack(pady=5)
    tk.Button(root, text="Back", width=get_button_width("Back"), command=lambda: show_operation_menu(dataset)).pack(pady=10)

def run_gradient_test_network(dataset):
    clear_window()

    tk.Label(root, text=f"Gradient Test on {dataset}", font=("Helvetica", 14, "bold")).pack(pady=10)

    # ----------- Console Output Section ----------- 
    output_frame = tk.LabelFrame(root, text="Console Output", padx=5, pady=5, font=("Helvetica", 10, "bold"))
    output_frame.pack(padx=10, pady=10, fill='x', expand=False)

    output_text = ScrolledText(output_frame, wrap=tk.WORD, height=15, width=100, font=("Courier", 9))
    output_text.pack(padx=5, pady=5, fill='both', expand=True)

    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget
        def write(self, s):
            self.text_widget.insert(tk.END, s)
            self.text_widget.see(tk.END)
        def flush(self):
            pass

    sys.stdout = StdoutRedirector(output_text)

    # ----------- Plot Output Section ----------- 
    plot_output_frame = tk.LabelFrame(root, text="Plot Output", padx=5, pady=5, font=("Helvetica", 10, "bold"))
    plot_output_frame.pack(padx=10, pady=10, fill='both', expand=True)

    canvas_frame = tk.Frame(plot_output_frame)
    canvas_frame.pack(padx=5, pady=5, fill='both', expand=True)

    canvas = tk.Canvas(canvas_frame)
    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollable_plot_frame = tk.Frame(canvas)

    scrollable_plot_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_plot_frame, anchor="center")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # ----------- Run Test ----------- 
    print(f"Running Gradient Test on model using {dataset}...\n")

    model = default_neural_network(dataset)
    X = datasets[dataset].train_data
    C = datasets[dataset].train_indicators

    gradient_test_network(model, X, C, num_samples=1, plot=True, parent_frame=scrollable_plot_frame)

    print("Finished - inspect the plot for gradient approximation errors.\n")

    sys.stdout = sys.__stdout__

    tk.Button(root, text="Back", width=20, command=lambda: show_tests_menu(dataset)).pack(pady=5)
    
def run_jacobian_test_network(dataset):
    clear_window()

    # Title label for the test
    tk.Label(root, text=f"Jacobian Test on {dataset}", font=("Helvetica", 14, "bold")).pack(pady=10)

    # ----------- Output Frame (Console Output) ----------- 
    output_frame = tk.LabelFrame(root, text="Console Output", padx=5, pady=5, font=("Helvetica", 10, "bold"))
    output_frame.pack(padx=10, pady=10, fill='x', expand=False)

    # Output text area with tight wrapping
    output_text = ScrolledText(output_frame, wrap=tk.WORD, height=15, width=100, font=("Courier", 9))
    output_text.pack(padx=5, pady=5, fill='both', expand=True)

    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, s):
            self.text_widget.insert(tk.END, s)
            self.text_widget.see(tk.END)

        def flush(self):
            pass

    sys.stdout = StdoutRedirector(output_text)  # Redirect print output

    # ----------- Plot Output Frame ----------- 
    plot_output_frame = tk.LabelFrame(root, text="Plot Output", padx=5, pady=5, font=("Helvetica", 10, "bold"))
    plot_output_frame.pack(padx=10, pady=10, fill='both', expand=True)

    # Scrollable plot area (same as before)
    canvas_frame = tk.Frame(plot_output_frame)
    canvas_frame.pack(padx=5, pady=5, fill='both', expand=True)

    canvas = tk.Canvas(canvas_frame)
    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollable_plot_frame = tk.Frame(canvas)

    scrollable_plot_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_plot_frame, anchor="center")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    # ---------------------------------------------

    # Run the Jacobian test with plotting
    print(f"Running Jacobian Test on model using {dataset}...\n")

    model = default_neural_network(dataset)  # Load default model
    sample_input = datasets[dataset].train_data

    # Run with plotting inside the scrollable plot frame
    jacobian_test_network(model, sample_input, sample_num=1, plot=True, parent_frame=scrollable_plot_frame)
    
    # Print completion message in the console output
    print(f"Finished - look at gradient plots for results\n")
    
    # Restore stdout after the test
    sys.stdout = sys.__stdout__  # Optional: Restore stdout if desired
    
    # Back button
    tk.Button(root, text="Back", width=20, command=lambda: show_tests_menu(dataset)).pack(pady=5)

def train_model(dataset, model_type, layers=None, lr=None, batch_size=None, default=False):
    clear_window()

    # === Title ===
    title = f"{'Default' if default else 'Custom'} Training - {model_type} on {dataset}"
    tk.Label(root, text=title, font=("Helvetica", 14, "bold")).pack(pady=10)

    # === Console Output ===
    output_frame = tk.LabelFrame(root, text="Console Output", padx=5, pady=5, font=("Helvetica", 10, "bold"))
    output_frame.pack(padx=10, pady=(5, 10), fill='x', expand=False)

    output_text = ScrolledText(output_frame, wrap=tk.WORD, height=15, width=100, font=("Courier", 9))
    output_text.pack(padx=5, pady=5, fill='both', expand=True)

    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, s):
            self.text_widget.insert(tk.END, s)
            self.text_widget.see(tk.END)

        def flush(self):
            pass

    sys.stdout = StdoutRedirector(output_text)
    sys.stderr = StdoutRedirector(output_text)

    # === Scrollable Plot Frame ===
    plot_frame = tk.LabelFrame(root, text="Training Plot", font=("Helvetica", 10, "bold"))
    plot_frame.pack(padx=10, pady=10, fill='both', expand=True)

    outer_frame = tk.Frame(plot_frame)
    outer_frame.pack(fill='both', expand=True)

    canvas = tk.Canvas(outer_frame)
    scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    scrollable_plot_frame = tk.Frame(canvas)

    scrollable_plot_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_plot_frame, anchor="center")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # === Back Button ===
    tk.Button(root, text="Back", width=20, command=lambda: show_train_options(dataset, model_type)).pack(pady=5)

    # === Training Thread ===
    def training_process():
        nonlocal lr, batch_size
        try:
            if default:
                if model_type == "NeuralNetwork":
                    model = default_neural_network(dataset)
                elif model_type == "ResNet":
                    model = default_resnet(dataset)
                else:
                    raise ValueError("Invalid model type.")
                lr = model.lr
                batch_size = model.optimizer.batch_size
                
            else:
                if not layers:
                    messagebox.showwarning("Input Error", "Please provide valid layers and hidden size for custom model.")
                 
                print(f"lr: {lr}, batch_size: {batch_size}")
                model = costumized_network(dataset, model_type, layers, lr, batch_size)

            print("Model Summary:")
            model.print_model()

            data = datasets[dataset]
            metrics = model.train(
                X_train=data.train_data,
                y_train=data.train_labels,
                C_train=data.train_indicators,
                epochs=200,
                batch_size=batch_size,
                X_val=data.test_data,
                y_val=data.test_labels,
                C_val=data.test_indicators
            )

            # Plot in main thread
            root.after(0, lambda: model.plot_metrics(metrics, parent=scrollable_plot_frame))
            sys.stdout = sys.__stdout__  # Restore when done

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred: {e}")
            sys.stdout = sys.__stdout__

    threading.Thread(target=training_process).start()

def least_example():
    np.random.seed(42)
    n, m = 3, 100   # Features, Samples
    
    X = np.random.rand(n, m)
    true_W = np.array([[1.0], [2.0], [3.0]])
    true_b = np.array([[0.5]])  
    y = (X.T @ true_W + true_b.T).reshape(-1, 1)
    
    # Initialize weights and bias
    initial_W = np.random.randn(n, 1)
    initial_b = np.random.randn(1, 1)

    # define optimizer
    optimizer = Optimizer(0.1, 100)
    updated_W, updated_b, losses, success_percentages, val_losses, val_success_percentages = optimizer.SGD(
        X=X,
        y=y,
        C=np.empty((0, m)),  # C not used in least squares
        W=initial_W,
        b=initial_b,
        loss = Loss(),
        epochs=200
    )
    print("True Weights:", true_W.flatten())
    print("Learned Weights:", updated_W.flatten())
    print("True Bias:", true_b.flatten())
    print("Learned Bias:", updated_b.flatten())

def run_experiment(dataset, exp_num):
    clear_window()

    # Title
    titles = {
        1: "Depth Experiment NN",
        2: "Depth Experiment ResNet",
        3: "Parameter Restriction Experiment NN"
    }
    tk.Label(root, text=f"{titles[exp_num]} - on {dataset}", font=("Helvetica", 14, "bold")).pack(pady=10)

    # Console Output
    output_frame = tk.LabelFrame(root, text="Console Output", padx=5, pady=5, font=("Helvetica", 10, "bold"))
    output_frame.pack(padx=10, pady=10, fill='x', expand=False)

    output_text = ScrolledText(output_frame, wrap=tk.WORD, height=15, width=100, font=("Courier", 9))
    output_text.pack(padx=5, pady=5, fill='both', expand=True)

    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, s):
            self.text_widget.insert(tk.END, s)
            self.text_widget.see(tk.END)

        def flush(self):
            pass

    sys.stdout = StdoutRedirector(output_text)

    # Plot Frame
    plot_frame = tk.LabelFrame(root, text="Plot Output", font=("Helvetica", 10, "bold"))
    plot_frame.pack(padx=10, pady=10, fill='both', expand=True)

    outer_frame = tk.Frame(plot_frame)
    outer_frame.pack(fill='both', expand=True)

    canvas = tk.Canvas(outer_frame)
    scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    scrollable_plot_frame = tk.Frame(canvas)

    scrollable_plot_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_plot_frame, anchor="center")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Back Button
    tk.Button(root, text="Back", width=20, command=lambda: show_experiment_menu(dataset)).pack(pady=5)

    # Threaded logic
    def run():
        if exp_num == 1:
            depth_experiment_NN(datasets[dataset], parent=scrollable_plot_frame)
        elif exp_num == 2:
            depth_experiment_RN(datasets[dataset], parent=scrollable_plot_frame)
        elif exp_num == 3:
            paramater_restriction_NN(datasets[dataset], parent=scrollable_plot_frame)
        sys.stdout = sys.__stdout__  # restore stdout when done

    threading.Thread(target=run).start()

def depth_experiment_NN(data, parent=None):
    # Hyperparameters
    lr = 0.01
    batch = 64
    epochs = 200
    activation = 'relu'
    input = data.train_data.shape[0]
    output = data.train_indicators.shape[1]
    hidden = 20

    # Define model depths
    network_configs = [
        ("0 Hidden Layers", [(input, output, 'softmax')]),
        ("1 Hidden Layer", [(input, hidden, activation), (hidden, output, 'softmax')]),
        ("2 Hidden Layers", [(input, hidden, activation), (hidden, hidden, activation), (hidden, output, 'softmax')]),
        ("6 Hidden Layers", [(input, hidden, activation)] + [(hidden, hidden, activation)] * 5 + [(hidden, output, 'softmax')]),
    ]

    results = {}

    print("Starting Depth Experiment for Neural Network")
    print(f"Using Hyperparameters -> Learning Rate: {lr}, Batch Size: {batch}, Epochs: {epochs}, Activation: {activation}")
    print("--------------------------------------------------")

    for name, config in network_configs:
        print(f"\nTraining configuration: {name}")
        model = NeuralNetwork(config, lr, batch)
        model.print_model()  # Print architecture
        print("Beginning training...\n")

        metrics = model.train(
            X_train=data.train_data,
            y_train=data.train_labels,
            C_train=data.train_indicators,
            epochs=epochs,
            batch_size=batch,
            X_val=data.test_data,
            y_val=data.test_labels,
            C_val=data.test_indicators
        )

        print(f"\nFinished training {name}")
        print("Generating plots...\n")
        
        results[name] = metrics
        
        # Schedule plotting in the main thread
        def plot_in_main_thread():
            model.plot_metrics(metrics, parent=parent)

        root.after(0, plot_in_main_thread)  # Schedule plot in main thread

    print("\nAll training runs completed.")
    return results

def paramater_restriction_NN(data, parent=None):
    # Define hyperparameters
    lr = 0.01
    batch = 32
    epochs = 200
    activation = 'tanh'
    input_dim = data.train_data.shape[0]
    output_dim = data.train_indicators.shape[1]

    # Define network configurations
    network_configs = [
        ("Shallow Network (1 Hidden Layer)", [(input_dim, 25, activation), (25, output_dim, 'softmax')]),  # 160 parameters
        ("Moderate Depth (2 Hidden Layers - Balanced)", [(input_dim, 9, activation), (9, 12, activation), (12, output_dim, 'softmax')]),  # 192 parameters
        ("Deeper Network (3 Hidden Layers - Balanced)", [(input_dim, 7, activation), (7, 8, activation), (8, 8, activation), (8, output_dim, 'softmax')]),  # 160 parameters
        ("Deeper Network (4 Hidden Layers - Balanced)", [
            (input_dim, 6, activation),  
            (6, 6, activation),  
            (6, 7, activation),  
            (7, 7, activation),  
            (7, output_dim, 'softmax')
        ]),
        ("Deeper Network (5 Hidden Layers - Balanced)", [
            (input_dim, 6, activation),  
            (6, 6, activation),  
            (6, 6, activation),  
            (6, 6, activation),  
            (6, 6, activation),  
            (6, output_dim, 'softmax')
        ]),
        ("Optimized 6-Layer Network", [
            (input_dim, 5, activation),  
            (5, 5, activation),  
            (5, 5, activation),  
            (5, 5, activation),  
            (5, 5, activation),  
            (5, 5, activation),  
            (5, output_dim, 'softmax')
        ]),
        ("Optimized 7-Layer Network", [
            (input_dim, 4, activation),  
            (4, 4, activation),  
            (4, 4, activation),  
            (4, 4, activation),  
            (4, 4, activation), 
            (4, 4, activation), 
            (4, 4, activation),  
            (4, output_dim, 'softmax')
        ])
    ]

    results = {}

    print("Starting Parameter Restriction Experiment (Neural Network)")
    print(f"Hyperparameters: Learning Rate = {lr}, Batch Size = {batch}, Epochs = {epochs}, Activation = '{activation}'")
    print("-" * 60)

    for name, config in network_configs:
        print(f"\nTraining Configuration: {name}")
        model = NeuralNetwork(config, lr, batch)
        model.print_model()  # If available
        print("Starting training...\n")

        metrics = model.train(
            X_train=data.train_data,
            y_train=data.train_labels,
            C_train=data.train_indicators,
            epochs=epochs,
            batch_size=batch,
            X_val=data.test_data,
            y_val=data.test_labels,
            C_val=data.test_indicators
        )

        print(f"Finished training: {name}")
        print("Plotting results...\n")
        
        results[name] = metrics
        
        # Schedule plotting in the main thread
        def plot_in_main_thread():
            model.plot_metrics(metrics, parent=parent)

        root.after(0, plot_in_main_thread)  # Schedule plot in main thread

    print("All configurations have been trained and evaluated.")
    return results

def depth_experiment_RN(data, parent=None):
    # Define hyperparameters
    lr = 0.01
    batch = 64
    epochs = 200
    activation = 'tanh'
    input_dim = data.train_data.shape[0]
    output_dim = data.train_indicators.shape[1]
    hidden = 20

    # Define network configurations
    network_configs = [
        ("0 Hidden Layers", [(input_dim, output_dim, 'softmax')]),
        ("1 Hidden Layer", [(input_dim, hidden, activation), (input_dim, output_dim, 'softmax')]),
        ("2 Hidden Layers", [(input_dim, hidden, activation), (input_dim, hidden, activation), (input_dim, output_dim, 'softmax')]),
        ("6 Hidden Layers", [(input_dim, hidden, activation)] * 6 + [(input_dim, output_dim, 'softmax')]),
        ("10 Hidden Layers", [(input_dim, hidden, activation)] * 10 + [(input_dim, output_dim, 'softmax')]),
        ("20 Hidden Layers", [(input_dim, hidden, activation)] * 20 + [(input_dim, output_dim, 'softmax')])
    ]

    results = {}

    print("Starting Depth Experiment for ResNet")
    print(f"Using Hyperparameters -> Learning Rate: {lr}, Batch Size: {batch}, Epochs: {epochs}, Activation: {activation}")
    print("--------------------------------------------------")

    for name, config in network_configs:
        print(f"\nTraining configuration: {name}")
        model = ResNet(config, lr, batch)
        model.print_model()  # Print architecture
        print("Beginning training...\n")

        metrics = model.train(
            X_train=data.train_data,
            y_train=data.train_labels,
            C_train=data.train_indicators,
            epochs=epochs,
            batch_size=batch,
            X_val=data.test_data,
            y_val=data.test_labels,
            C_val=data.test_indicators
        )

        print(f"\nFinished training {name}")
        print("Generating plots...\n")

        results[name] = metrics
        # Schedule plotting in the main thread
        def plot_in_main_thread():
            model.plot_metrics(metrics, parent=parent)

        root.after(0, plot_in_main_thread)  # Schedule plot in main thread
    
    print("\nAll training runs completed.")
    return results

def find_best_hyperparameters_sgd(Data, epochs, learning_rates=None, batch_sizes=None, plot=True, parent_frame=None):
    if learning_rates is None:
        learning_rates = [0.001, 0.01, 0.1, 0.5]
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 200, 256]

    best_accuracy = 0
    best_lr = None
    best_batch_size = None

    best_train_losses = []
    best_train_success_percentages = []
    best_val_losses = []
    best_val_success_percentages = []

    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            optimizer = Optimizer(lr, bs)
            print(f"Testing hyperparameters: Learning Rate = {lr}, Batch Size = {bs}")
            W, b, train_losses, train_success_percentages, val_losses, val_success_percentages = optimizer.SGD(
                X=Data.train_data,
                y=Data.train_labels,
                C=Data.train_indicators,
                W=Data.weights_train,
                b=Data.bias_train,
                loss=Loss(),
                epochs=epochs,
                X_val=Data.test_data,
                y_val=Data.test_labels,
                C_val=Data.test_indicators,
                plot=False
            )

            if val_success_percentages[-1] > best_accuracy:
                best_accuracy = val_success_percentages[-1]
                best_lr = lr
                best_batch_size = bs
                best_train_losses = train_losses
                best_train_success_percentages = train_success_percentages
                best_val_losses = val_losses
                best_val_success_percentages = val_success_percentages

            results.append({
                "learning_rate": lr,
                "batch_size": bs,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_success_percentages,
                "val_accuracies": val_success_percentages,
            })

    print(f"Best hyperparameters: Learning Rate = {best_lr}, Batch Size = {best_batch_size}")

    # Always define fig and ax
    fig, ax = None, None

    if plot:
        # Schedule plotting in the main thread
        def plot_in_main_thread():
            nonlocal fig, ax
            for result in results:
                if result["learning_rate"] == best_lr and result["batch_size"] == best_batch_size:
                    optimizer.plot_sgd_results(result["train_losses"], result["train_accuracies"], 
                                               "Training Loss and Accuracy", parent_frame=parent_frame)
                    optimizer.plot_sgd_results(result["val_losses"], result["val_accuracies"], 
                                               "Validation Loss and Accuracy", parent_frame=parent_frame)

            labels = [f"lr={res['learning_rate']}, bs={res['batch_size']}" for res in results]
            final_val_accuracies = [res['val_accuracies'][-1] for res in results]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(labels, final_val_accuracies)
            ax.set_xlabel("Learning Rate and Batch Size")
            ax.set_ylabel("Final Validation Accuracy")
            ax.set_title("Final Validation Accuracy for Different Hyperparameters")

            positions = range(len(labels))
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right")

            plt.tight_layout()

            if parent_frame:
                canvas_plot = FigureCanvasTkAgg(fig, master=parent_frame)
                canvas_plot.draw()
                canvas_plot.get_tk_widget().pack(pady=10)
            else:
                plt.show()

        # Schedule plot function to run on the main thread
        root.after(0, plot_in_main_thread)

    return fig, ax, results

def find_best_hyperparams(dataset):
    clear_window()  # Clear existing GUI content on the root window

    # Title
    tk.Label(root, text=f"Hyperparameter Search - {dataset}", font=("Helvetica", 14, "bold")).pack(pady=10)

    # Console Output
    console_frame = tk.LabelFrame(root, text="Console Output", padx=5, pady=5, font=("Helvetica", 10, "bold"))
    console_frame.pack(padx=10, pady=(10, 5), fill='x')

    console = ScrolledText(console_frame, wrap=tk.WORD, height=15, width=100, font=("Courier", 9))
    console.pack(padx=5, pady=5, fill='both', expand=True)

    class ConsoleRedirect:
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, msg):
            def append():
                if self.text_widget.winfo_exists():
                    self.text_widget.insert(tk.END, msg)
                    self.text_widget.see(tk.END)
            root.after(0, append)

        def flush(self):
            pass

    original_stdout = sys.stdout
    sys.stdout = ConsoleRedirect(console)

    # Plot Output
    plot_frame = tk.LabelFrame(root, text="Plot Output", font=("Helvetica", 10, "bold"))
    plot_frame.pack(padx=10, pady=5, fill='both', expand=True)

    outer_frame = tk.Frame(plot_frame)
    outer_frame.pack(fill='both', expand=True)

    canvas = tk.Canvas(outer_frame)
    scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    scrollable_plot_frame = tk.Frame(canvas)

    scrollable_plot_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_plot_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Back Button
    tk.Button(root, text="Back", width=20, command=lambda: show_operation_menu(dataset)).pack(pady=5)

    # Training Thread
    def run():
        try:
            Data = datasets[dataset]
            epochs = 20
            print(f"Finding best hyperparameters for {dataset} over {epochs} epochs...\n")

            for widget in scrollable_plot_frame.winfo_children():
                widget.destroy()

            fig, ax, results = find_best_hyperparameters_sgd(
                Data, epochs=epochs, plot=True, parent_frame=scrollable_plot_frame
            )

            if not results:
                print("No results returned from hyperparameter search.")
                return

            # Filter out None entries just in case
            valid_results = [r for r in results if r is not None and 'val_accuracies' in r and r['val_accuracies']]
            if not valid_results:
                print("No valid results to evaluate.")
                return

            best = max(valid_results, key=lambda r: r['val_accuracies'][-1])
            print(f"\nBest Learning Rate: {best['learning_rate']}")
            print(f"Best Batch Size: {best['batch_size']}")

        except Exception as e:
            print(f"An error occurred during hyperparameter search:\n{e}")

        finally:
            sys.stdout = original_stdout

    threading.Thread(target=run).start()


def main():
    
    show_start_menu()
    root.mainloop()
    
    

if __name__ == "__main__":
    main()