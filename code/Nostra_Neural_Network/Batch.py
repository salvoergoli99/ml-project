from  Nostra_Neural_Network.LossFunction import LossFunction
import numpy as np
import random
import sys
#VA FATTO LO SHUFFLE DEL DATA SET OGNI EPOCA O POCO PIù PERCHè SI POTREBBERO FORMARE DEI CICLI
#OPPURE SI PUO FARE UNO SHUFFLE DELL'ORDINE IN CUI VENGONO PRESI I MINI BATCH
class Batch:
    def __init__(self,X,Y,Loss_Function="MSE"):
        """
        Initialize the Batch class with data and loss function.

        Parameters:
        X (list): List of feature sets.
        Y (list): List of label sets.
        Loss_Function (str): Type of loss function to use. Default is "MSE".
        """
        self.current_state = 0  # Initialize the current state index
        self.lunghezza = len(X)  # Length of the dataset
        
        # Initialize lists for features, labels, and loss functions
        self.X_list = X
        self.Y_list = Y
        self.Loss_list = [LossFunction(Loss_Function, y) for y in Y]
        
        # Set the current feature, label, and loss function
        self.X = self.X_list[0]
        self.Y = self.Y_list[0]
        self.Loss = self.Loss_list[0]
        
    def next(self):
        """
        Advance to the next mini-batch, shuffling the dataset each epoch

        Returns:
        None
        """
        self.current_state += 1  # Increment the current state index
        
        if self.current_state % self.lunghezza == 0:
            # Shuffle the dataset at the end of each epoch
            seed = random.random()  # Generate a random seed
            random.seed(seed)
            random.shuffle(self.X_list)
            random.seed(seed)
            random.shuffle(self.Y_list)
            random.seed(seed)
            random.shuffle(self.Loss_list)
        
        # Update the current feature, label, and loss function
        self.X = self.X_list[self.current_state % self.lunghezza]
        self.Y = self.Y_list[self.current_state % self.lunghezza]
        self.Loss = self.Loss_list[self.current_state % self.lunghezza]
        
