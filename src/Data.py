import numpy as np
import scipy.io as sio


class Data:
    def __init__(self, file_path, name):
        self.name = name
        self.data = self.load_data(file_path)
        
        preprocess = self.preprocess_data()
        
        self.train_data = preprocess[0]
        self.test_data = preprocess[1]
        self.train_indicators = preprocess[2]
        self.test_indicators = preprocess[3]
        self.train_labels = preprocess[4]
        self.test_labels = preprocess[5]
        self.weights_train = preprocess[6]
        self.bias_train = preprocess[7]
        

    def load_data(self, file_path):
        """
        Load data from a .mat file.

        Parameters:
        file_path (str): Path to the .mat file.

        Returns:
        dict: Loaded data.
        """
        return sio.loadmat(file_path)

    def preprocess_data(self):
        
        # Data - X matrix (n x m)
        train_data = self.data['Yt']
        test_data = self.data['Yv']
        
        # Indicators - C matrix (m x l)
        train_indicators = self.data['Ct'].T  # Transpose to match the expected shape
        test_indicators = self.data['Cv'].T  # Transpose to match the expected shape
        
        # Labels - y vector (m, 1)
        train_labels = np.argmax(train_indicators, axis=1).reshape(-1, 1)
        test_labels = np.argmax(test_indicators, axis=1).reshape(-1, 1)
        
        # train_labels = np.argmax(train_indicators, axis=1)
        # test_labels = np.argmax(test_indicators, axis=1)
        
        # Weights - W matrix (n x l)
        weights_train = np.random.rand(train_data.shape[0], train_indicators.shape[1])
        
        # Bias - b vector (l, 1)
        bias_train = np.random.rand(train_indicators.shape[1], 1)
        
        return train_data, test_data, train_indicators, test_indicators, train_labels, test_labels, weights_train, bias_train

    def print_shapes(self):

        print(f"-----{self.name}-----")
        
        #print n m l
        print(f"n: {self.train_data.shape[0]}")
        print(f"m: {self.train_data.shape[1]}")
        print(f"l: {self.train_indicators.shape[1]}")
        
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Train indicators shape: {self.train_indicators.shape}")
        print(f"Test indicators shape: {self.test_indicators.shape}")
        print(f"Train labels shape: {self.train_labels.shape}")
        print(f"Test labels shape: {self.test_labels.shape}")
        print(f"Train weights shape: {self.weights_train.shape}")
        print(f"Train bias shape: {self.bias_train.shape}")
        print("------------------")
        print()
