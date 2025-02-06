import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from  Nostra_Neural_Network.Neural_Network import *
import  Nostra_Neural_Network.Neural_Network as NN
import copy
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from  Nostra_Neural_Network.K_FOLD import k_fold

class Average:
    """
    This class implements an ensemble learning approach using averaging.

    It trains a neural network multiple times with different initializations and takes the average of the predictions as the final output.
    """
    def __init__(self,nn, X_TR, Y_TR, X_TS, epoche, epoche_prova):
        """
        Initializes the Average ensemble with a neural network model, training and validation data, test data, training epochs, and validation epochs.

        Args:
            nn (object): The Neural Network model to be used for training.
            X_TR (np.array): Training data features.
            Y_TR (np.array): Training data targets.
            X_TS (np.array): Test data features.
            epoche (int): Number of training epochs for the final training stage.
            epoche_prova (int): Number of training epochs for chooding the best inizialization .
        """
        self.X_TS = X_TS    # Store test data for prediction
        
        # Perform 10-fold cross-validation to average predictions across folds
        fold = k_fold(10, X_TR, Y_TR)
        
        self.nn = nn    # Store the neural network model
        y_mean=np.zeros((self.X_TS.shape[0],3))  #Initialize array to store average predictions
        
        while fold.EO():
            # Get data for current fold
            self.X_TR, self.Y_TR, self.X_VL, self.Y_VL = fold.data()
            
            # Train the model and get predictions on test data for this fold
            y_pred = self.train(epoche, epoche_prova)
            y_mean += y_pred
            
        # Calculate the final average prediction by dividing by the number of folds
        self.avg = y_mean/10
    def predict(self):
        """
        Returns the average prediction of the ensemble.

        This method simply returns the pre-computed average prediction stored in the `self.avg` attribute.
        """
        return self.avg
    def train(self,epoche,epoche_prova = 7000):
        """
        Trains the neural network with dynamic learning rate and finds the best configuration.

        This method performs the following steps:
            1. Runs multiple training iterations with different random initializations.
            2. Finds the iteration with the lowest validation Mean Absolute Error (MAE).
            3. Trains the model again using the best configuration for the specified number of epochs.
            4. Returns the final prediction on the test data using the trained model.

        Args:
            epoche (int): Number of training epochs for the final training stage.
            epoche_prova (int): Number of training epochs for hyperparameter tuning (validation stage).

        Returns:
            np.array: The final prediction on the test data.
        """
        nn = self.nn
        layers_min = []  # Stores the best model layers configuration
        minimo = 100000  # Initialize minimum validation MAE
        
        for i in range(5):
            # Set the network to train mode with current fold data
            nn.set_train(self.X_TR, self.Y_TR)

            # Initialize weights with Xavier uniform distribution
            nn.Inizialization(type="Xavier_uniform",scale=1)

            # Train the model with dynamic learning rate using Train object
            train = Train(self.nn, self.X_TR, self.Y_TR, self.X_VL, self.Y_VL)
            learning_rate = train.dynamic_learn(step = 10, epoche = epoche_prova, max_prove=50,scaling = 0.90, verbouse = False)
            
             # Update minimum validation MAE and best configuration if necessary
            if (train.MEE_VL[-1] <=minimo):
                minimo = train.MEE_VL[-1]
                layers_min = copy.deepcopy(nn.layers)

        # Train the model again with the best configuration for the final training stage
        train=Train(nn, self.X_TR, self.Y_TR, self.X_VL, self.Y_VL)
        nn.layers = layers_min
        learning_rate_zero = learning_rate(epoche_prova - 1)    #take the learning rate used in the last iteration
        learning_rate = train.dynamic_learn(step = 10, epoche = epoche, max_prove=50,scaling = 0.90, verbouse = False, learning_rate_init = learning_rate_zero)
        print(" Test Mean Euclidian Error on the fold out of training:",train.MEE_VL[-1])
        return nn.forward(self.X_TS)

