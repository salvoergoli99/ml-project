import numpy as np
class layer:
    def __init__(self,out_unit,in_unit, adam = False):
        """
        Initialize the layer with the given number of output and input units.
        
        Parameters:
        out_unit (int): Number of neurons in the current layer.
        in_unit (int): Number of neurons in the previous layer.
        adam (bool): Whether to use Adam optimization. Default is False.
        """
        # Number of neurons in the output and input layers
        self.out_unit = out_unit
        self.in_unit = in_unit

        # Weight matrix, gradients, and bias initialization
        self.W = None  # Weight matrix (out_unit x in_unit)
        self.grad_W = 0  # Gradient of the weight matrix (out_unit x in_unit)
        self.bias = None  # Bias vector (out_unit,)
        self.grad_bias = 0  # Gradient of the bias vector (out_unit,)

        # Additional attributes
        self.activation_function = None  # Placeholder for activation function
        self.u = None  # Placeholder for pre-activation values
        self.net = None  # Placeholder for post-activation values
        self.delta = None  # Placeholder for backpropagation deltas

        # Adam optimizer parameters
        if adam:
            self.m_grad_W = np.zeros((self.out_unit, self.in_unit))  # First moment estimate for weights
            self.m_grad_bias = np.zeros(self.out_unit)  # First moment estimate for biases
            self.v_grad_W = np.zeros((self.out_unit, self.in_unit))  # Second moment estimate for weights
            self.v_grad_bias = np.zeros(self.out_unit)  # Second moment estimate for biases
        
    def inizialization(self,type,std_dev=1,limit=(0,1)):
        """
        Initialize the weights and biases of the layer.
        
        Parameters:
        type (str): Type of initialization ('normal' or 'uniform').
        std_dev (float): Standard deviation for normal distribution. Default is 1.
        limit (tuple): Limits for uniform distribution. Default is (0, 1).
        
        Returns:
        None
        """
        match type:
            case "normal":
                # Initialize weights with normal distribution
                self.W = std_dev * np.random.randn(self.out_unit, self.in_unit)
                self.bias = np.zeros(self.out_unit)  # Initialize biases to zero
            case "uniform":
                # Initialize weights with uniform distribution
                self.W = np.random.uniform(limit[0], limit[1], (self.out_unit, self.in_unit))
                self.bias = np.zeros(self.out_unit)  # Initialize biases to zero

