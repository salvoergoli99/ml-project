import numpy as  np
from  Nostra_Neural_Network.layer import *
import random
from  Nostra_Neural_Network.Train import *
import copy
from  Nostra_Neural_Network.LossFunction import *
from  Nostra_Neural_Network.Batch import *
from  Nostra_Neural_Network.Activation_Function import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from  Nostra_Neural_Network.metrics import *

class Neural_Network:
    def __init__(self, out_dim, in_dim, adam = False):
        """
        Initialize the Neural Network with the input and output dimensions.

        Parameters:
        out_dim (int): Number of output units.
        in_dim (int): Number of input units.
        adam (bool): Whether to use Adam optimization algorithm. Default is False.
        """
        # Create the initial layer of the neural network
        self.layers = [layer(out_dim, in_dim, adam)]
        # Set activation function for the initial layer to identity
        self.layers[0].activation_function = activation_function("identity")
        
        self.LossFunction = None
        
        self.gradient_treshold = -1
        self.lamda_1 = 0
        self.lamda_2 = 0
        self.alfa = 0
        self.learning_rate = None
        
        self.adam = adam
        # ADAM SETTINGS
        if self.adam:
            self.beta_1 = 0.9       # Constant for first-order moment
            self.beta_2 = 0.999     # Constant for second-order moment
            self.delta = 1e-8       # Constant for numerical stabilization
            self.step = 0

    def add_layer(self, dim_layer, activation = "tanh"):
        """
        Add a new layer to the neural network.

        Parameters:
        dim_layer (int): Number of units in the new layer.
        activation (str): Activation function for the new layer. Default is "tanh".
        """
        # Remove the last layer from the network
        last_layer = self.layers.pop()
        # Create and append the new layer with dimensions compatible with the previous layer
        new_layer = layer(last_layer.out_unit, dim_layer, self.adam)
        new_layer.activation_function = last_layer.activation_function
        self.layers.append(new_layer)
        
        # Add the new layer with specified dimensions
        last_layer = layer(dim_layer, last_layer.in_unit, self.adam)
        last_layer.activation_function = activation_function(activation)
        self.layers.append(last_layer)
    
    def Inizialization(self, type, scale):
        """
        Initialize the weights of the neural network using specified initialization method.

        Parameters:
        type (str): The type of initialization method.
        scale (float): Scaling factor for initialization.
        """
        match type:
            case "He":
                for lay in self.layers:
                    std_dev = np.sqrt(2 / lay.in_unit)
                    lay.inizialization("normal", std_dev=std_dev * scale)
            case "Xavier_normal":
                out = 0
                for lay in self.layers:
                    std_dev = np.sqrt(2 / (out + lay.in_unit))
                    out = lay.in_unit
                    lay.inizialization("normal", std_dev=std_dev * scale)
            case "Xavier_uniform":
                out = 0
                for lay in self.layers:
                    delta = scale * np.sqrt(6 / (out + lay.in_unit))
                    out = lay.in_unit
                    lay.inizialization("uniform", limit=(-delta, delta))
 
    def forward(self, input):
        """
        Perform forward pass through the neural network.

        Parameters:
        input (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output of the neural network.
        """
        x = input
        for lay in reversed(self.layers):
            lay.net = np.dot(x, lay.W.T) + lay.bias
            if np.isinf(lay.net).any():
                print("net inf")
            lay.u = lay.activation_function.function(lay.net)
            x = lay.u
        return x

    def delta_backprop(self):
        """
        Perform backpropagation to calculate the delta values for each layer.
        """
        self.layers[0].delta = self.LossFunction.gradient(self.layers[0].u) * self.layers[0].activation_function.gradient(self.layers[0].net)
        
        for i in range(1,len(self.layers)):
            self.layers[i].delta = self.layers[i].activation_function.gradient(self.layers[i].net) * np.dot(self.layers[i-1].delta,self.layers[i-1].W,)
    
    def step_train(self, learning_rate):
        """
        Perform one step of training using backpropagation and update weights and biases.

        Parameters:
        learning_rate (float): The learning rate for updating weights and biases.
        """
        # Set the loss function to the one used by the current batch
        self.LossFunction = self.batch.Loss
        
        # Perform forward pass through the neural network
        self.forward(self.batch.X)
        
        # Calculate delta values for backpropagation
        self.delta_backprop()
        
        next_u = self.batch.X
        for lay in reversed(self.layers):
            # Compute gradients for weights
            lay.grad_W = (1-self.alfa)* np.einsum('ki,kj->ij',lay.delta, next_u)/len(self.batch.X) + self.alfa*lay.grad_W
            next_u = lay.u
            
            # Compute gradients for biases
            lay.grad_bias = (1- self.alfa)*np.mean(lay.delta,axis=0)+self.alfa*lay.grad_bias 

            # Clip gradients if gradient threshold is set
            if self.gradient_treshold != -1:
                lay.grad_bias = np.clip(lay.grad_bias, a_min = -self.gradient_treshold, a_max = self.gradient_treshold)
                lay.grad_W = np.clip(lay.grad_W, a_min = -self.gradient_treshold, a_max = self.gradient_treshold)
            
            # Compute updates for weights and biases
            grad_W = lay.grad_W
            delta_bias = learning_rate*lay.grad_bias
            
            # Perform Adam optimization if enabled
            if self.adam:
                self.step+=1
                lay.m_grad_W = self.beta_1*lay.m_grad_W + (1-self.beta_1)*lay.grad_W
                lay.v_grad_W = self.beta_2*lay.v_grad_W + (1-self.beta_2)*np.square(lay.grad_W)
                lay.m_grad_bias = self.beta_1*lay.m_grad_bias + (1-self.beta_1)*lay.grad_bias
                lay.v_grad_bias = self.beta_2*lay.v_grad_bias + (1-self.beta_2)*np.square(lay.grad_bias)
            
                m_W_corrected = lay.m_grad_W / (1 - self.beta_1**self.step)
                v_W_corrected= lay.v_grad_W / (1 - self.beta_2**self.step)
                m_bias_corrected = lay.m_grad_bias / (1 - self.beta_1**self.step)
                v_bias_corrected = lay.v_grad_bias / (1 - self.beta_2**self.step)
                
                grad_W = m_W_corrected / (np.sqrt(v_W_corrected) + self.delta)
                delta_bias = learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + self.delta)

            # Update weights and biases
            delta_W = learning_rate*(grad_W + lay.W*self.lamda_2 + self.lamda_1*np.sign(lay.W))
            lay.W = lay.W - delta_W
            lay.bias = lay.bias - delta_bias
        
        # Move to the next minibatch
        self.batch.next()
            
    def set_train(self, X, Y, batch_size = -1, Loss_Function = "MSE", random_state = None):
        """
        Set up the training process with the given data.

        Parameters:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): Target labels.
        batch_size (int): Size of the mini-batches. Default is -1 (no batching).
        Loss_Function (str): The loss function to use for training. Default is "MSE".
        random_state (int): Random seed for reproducibility.
        """
        
        np.random.seed(random_state)
        
        
        # Handle cases where X and Y may be 1D arrays
        if len( X.shape) == 1:
            X = X.reshape((len(X),1))
            
        if len(Y.shape) == 1:
            Y = Y.reshape((len(Y),1))
        
        # Split data into mini-batches
        if batch_size != -1:
            X_split = [X[ i : i + batch_size] for i in range(0, X.shape[0], batch_size)]
            Y_split = [Y[ i : i + batch_size] for i in range(0, Y.shape[0], batch_size)]
        else : 
            X_split = [X]
            Y_split = [Y]
        
        # Create Batch object for training
        self.batch=Batch(X_split,Y_split,Loss_Function)
        self.batch_size = batch_size
  