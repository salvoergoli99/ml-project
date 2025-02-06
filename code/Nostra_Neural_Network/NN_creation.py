
#CLASSE CHE PRENDE UN DIZIONARIO E CREA UNA RETE NEURALE FATTA CON QUEI PARAMETRI
from Nostra_Neural_Network.Neural_Network import *
from  Nostra_Neural_Network.Activation_Function import activation_function
from  Nostra_Neural_Network.Batch import Batch

from  Nostra_Neural_Network.Train import Train
class NN_creation:
    def __init__(self,dic):
        # If the dictionary is empty, set default values and return None
        if dic == {}:
            self.dic={"lamda_1":0,"lamda_2":0,"alfa":0,"beta_1":0.9,"beta_2":0.99,"X_TR":None,"Y_TR":None,"X_TS":None,"Y_TS":None,"NN_shape":(3,10),"N_layer":1,"N_units":128,"activation_out":"identity","activation_hidden":"tanh","adam":False,"batch_size":-1,"Loss_function":"MSE","random_state":None,"gradient_treshold":10,"Inizialization_type": "Xavier_uniform","Inizialization_scale":1}
            return None
        
        # Set the class attributes using the values from the dictionary
        self.X_TR = dic["X_TR"]
        self.Y_TR = dic["Y_TR"]
        self.X_TS = dic["X_TS"]
        self.Y_TS = dic["Y_TS"]
        self.NN_shape = dic["NN_shape"]
        self.activation_out = dic["activation_out"]
        self.activation_hidden = dic["activation_hidden"]
        self.adam = dic["adam"]
        self.N_layers = dic["N_layer"]
        self.N_units = dic["N_units"]
        self.batch_size = dic["batch_size"]
        self.Loss_function = dic["Loss_function"]
        self.random_state = dic["random_state"]
        self.gradient_treshold = dic["gradient_treshold"]
        self.Inizialization_type = dic["Inizialization_type"]
        self.Inizialization_scale= dic["Inizialization_scale"]
        self.lamda_1 = dic["lamda_1"]
        self.lamda_2 = dic["lamda_2"]
        self.alfa = dic["alfa"]
        self.beta_1 = dic["beta_1"]
        self.beta_2 = dic["beta_2"]
        
        # Create an instance of the Neural_Network class using the provided parameters
        self.nn= Neural_Network(self.NN_shape[0],self.NN_shape[1],adam=self.adam)
        self.nn.layers[0].activation_function = activation_function(self.activation_out)
        
        # Add hidden layers based on the specified number
        for _ in range(self.N_layers):
            self.nn.add_layer(self.N_units,activation= self.activation_hidden)

        # Set the neural network training with the provided data
        self.nn.set_train(self.X_TR,self.Y_TR,batch_size=self.batch_size,Loss_Function = self.Loss_function,random_state=self.random_state)
        self.nn.gradient_treshold = self.gradient_treshold
        
        # Initialize the neural network weights
        self.nn.Inizialization(type=self.Inizialization_type,scale = self.Inizialization_scale)
        self.nn.lamda_1 = self.lamda_1
        self.nn.lamda_2 = self.lamda_2
        self.nn.alfa = self.alfa

        # Set parameters specific to the Adam optimizer, if used
        if self.adam:
            self.nn.beta_1 = self.beta_1
            self.nn.beta_2 = self.beta_2

    def automatic_learning(self,epoche,scaling=0.9,verbouse=False):
        """
        Performs learning of the neural network with a dynamic learning rate.
        
        Parameters:
        - epoche (int): Number of training epochs.
        - scaling (float): Learning rate reduction factor.
        - verbouse (bool): If True, prints training progress.
        
        Returns:
        - Tuple: Containing MSE_TR, MSE_VL, MEE_TR, MEE_VL, MEE_VL_std.
        """
        # Create an instance of the Train class for training
        train = Train(self.nn, self.X_TR, self.Y_TR, self.X_TS, self.Y_TS)
        
        # Perform dynamic learning rate for training
        learning_rate = train.dynamic_learn(step = 10, epoche = epoche, max_prove=100,scaling = scaling, verbouse = verbouse)
        
        # Return the training results
        return train.MSE_TR,train.MSE_VL,train.MEE_TR,train.MEE_VL,train.MEE_VL_std
        

if __name__=="__main__":
    nn= NN_creation({})
    print(nn.dic)
    dic =nn.dic
    dic["alfa"]=0.5
    dic[""]
        
