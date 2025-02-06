import numpy as np
class activation_function:
    def __init__(self,type):
        """
        Initialize the activation function with the specified type.

        Parameters:
        type (str): The type of activation function. Options are 'tanh', 'sigmoid', 
                    'identity', 'relu', and 'leaky_relu'.

        Attributes:
        self.function: The activation function chosen based on the type.
        self.gradient: The derivative of the chosen activation function
        """
        match type:
            case "tanh":
                self.function=np.tanh
                self.gradient = lambda x: 1 - np.tanh(x)**2
            case "sigmoid":
                self.function=lambda x: 1/(1+np.exp(-x))
                self.gradient= lambda x: self.function(x)*(1-self.function(x))
            case "identity":
                self.function=lambda x:x
                self.gradient=lambda x: 1
            case "relu":
                self.function=lambda x: x*(1*(x>0))
                self.gradient=lambda x: 1*(x>0)
            case "leaky_relu": 
                self.function=lambda x: x*(x>0)+0.01*x*(x<=0)
                self.gradient=lambda x: (x>0)+0.01*(x<=0)

