import numpy as np
class k_fold:
    
    def __init__(self,k,X,Y):
        """
        Initialize the k_fold class with the number of folds, and the data.
        
        Parameters:
        k (int): Number of folds.
        X (numpy.ndarray): Features dataset.
        Y (numpy.ndarray): Labels dataset.
        """
        self.X = X  # Features dataset
        self.Y = Y  # Labels dataset
        self.k = k  # Number of folds
        
        # Calculate the size of each fold
        step = len(X) // k
        
        # Split the dataset into k folds
        self.X_list = [X[step * i: step * (i + 1)] for i in range(k)]
        self.Y_list = [Y[step * i: step * (i + 1)] for i in range(k)]
        
        # Initialize the current state (fold) index
        self.current_state = 0
        
    def data(self):
        """
        Get the training and validation data for the current fold.
        
        Returns:
        tuple: A tuple containing training features (X_TR), training labels (Y_TR),
               validation features (X_VL), and validation labels (Y_VL).
        """
        # Reset current state if it exceeds the number of folds
        if self.current_state == len(self.X_list):
            self.current_state = 0
        
        # Create training data by stacking all folds except the current one
        X_TR = np.vstack([self.X_list[i] for i in range(len(self.X_list)) if i != self.current_state])
        Y_TR = np.vstack([self.Y_list[i] for i in range(len(self.X_list)) if i != self.current_state])
        
        # Validation data is the current fold
        X_VL = self.X_list[self.current_state]
        Y_VL = self.Y_list[self.current_state]
        
        # Move to the next fold
        self.current_state += 1
        
        return X_TR, Y_TR, X_VL, Y_VL

    def EO(self):
        """
        Check if all folds have been used for validation.
        
        Returns:
        int: 0 if all folds have been used, otherwise 1.
        """
        if self.current_state == len(self.X_list):
            self.current_state = 0
            return 0
        return 1
    