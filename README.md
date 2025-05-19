# Neural Networks from Scratch

This project implements neural networks from scratch using only NumPy. It includes custom implementations of core components like layers, activation functions, loss functions, optimizers, and full training pipelines for both standard neural networks and ResNets.

The project is designed for educational use, helping you understand the internals of deep learning systems through hands-on Python code.

This project also presents user friendly GUI - for easy external use.
---

## Project Structure

```
Neural-Networks-main/
│
├── data/                  # datasets (SwissRoll, Peaks, GMM)
├── images/                
├── src/                   # Core neural network implementation
│   ├── Activation.py      # Activation functions (ReLU, Softmax, etc.)
│   ├── Data.py            # Dataset loading and preprocessing
│   ├── NeuralNetwork.py   # Feedforward neural network class
│   ├── ResNet.py          # Custom residual network implementation
│   ├── layer.py           # Layer base class 
│   ├── loss.py            # Cross-entropy loss and Least Squares loss
│   ├── optimizer.py       # Optimizer (SGD, etc.)
│   ├── main.py            # Example training script
│   └── tests.py           # Unit tests and validation utilities
│
├── requirements.txt       # Python package requirements
└── README.md              # Project overview
```

---

## Features

-  Build and train custom feedforward neural networks and ResNets
-  Load and classify synthetic datasets: Swiss Roll, Peaks, GMM
-  Custom activation, loss, and optimizer modules
-  Designed for transparency and educational value
-  Minimal external dependencies (only NumPy and Matplotlib)
-  GUI for user friendly use

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Tal-Weiss/Neural-Networks.git
cd Neural-Networks
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the demo

```bash
python src/main.py
```

This will run training and evaluation code for the included datasets using your custom neural network implementation.

---

## Datasets

Included in the `data/` folder are three synthetic datasets:

- **Swiss Roll**
- **Peaks**
- **GMM (Gaussian Mixture Model)**

These are `.mat` files used for classification tasks and benchmarking different model architectures.

---

## License

This project is open-source and available under the MIT License.

---

## Contributing

Feel free to open issues or pull requests for bug fixes, new features, or improvements! This is an educational project — questions are welcome.

---

## Author

Created by **Tal Weiss** – feel free to connect on GitHub!
