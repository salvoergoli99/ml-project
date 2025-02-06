import copy
import  Nostra_Neural_Network.Neural_Network as NN
import numpy as np
from  Nostra_Neural_Network.LossFunction import *
from  Nostra_Neural_Network.Batch import Batch
from  Nostra_Neural_Network.Activation_Function import activation_function
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from  Nostra_Neural_Network.metrics import *
from sys import exit 
class Train:
    def static_learn(self, learning_rate, epoche, verbouse = True):
        """
        Train the neural network using static learning rate.

        Parameters:
        learning_rate (float or function): The learning rate or a function that returns the learning rate.
        epochs (int): Number of training epochs.
        verbose (bool): If True, print training progress.

        Returns:
        None
        """
        
        # If learning_rate is not a function, convert it into a function
        if type(learning_rate) != type(lambda x:x):
            temp = learning_rate
            learning_rate = lambda x: temp
            
        
        nn = self.Neural_Network
        iterazioni = epoche
        lunghezza_epoche= 1
        if nn.batch_size != -1:
            lunghezza_epoche = nn.batch.X_list
        iterazioni =epoche *lunghezza_epoche
        
        for i in range(iterazioni):
            # Perform a training step with the current learning rate
            nn.step_train(learning_rate(i))
            
            # Compute and store training and validation metrics
            self.MSE_TR.append(mean_squared_error(self.Y_TR, nn.forward(self.X_TR)))
            self.MSE_TS.append( mean_squared_error(self.Y_TS, nn.forward(self.X_TS)) )
            self.MEE_TR.append(MEE(self.Y_TR, nn.forward(self.X_TR)))
            self.MEE_TS.append( MEE(self.Y_TS, nn.forward(self.X_TS)) )
            self.MEE_VL.append(MEE(self.Y_VL, nn.forward(self.X_VL)))
            self.MEE_TS_std.append( MEE_std(self.Y_TS, nn.forward(self.X_TS)) )
            self.MEE_VL_std.append(MEE_std(self.Y_VL, nn.forward(self.X_VL)))
            self.MSE_VL.append( mean_squared_error(self.Y_VL, nn.forward(self.X_VL)))
            
            if verbouse:
                print(i//lunghezza_epoche,"Epoca     TR/TS MSE ", self.MSE_TR[i],"/",self.MSE_TS[i], "   TR/TS MEE ", self.MEE_TR[i],"/", self.MEE_TS[i] ," +- ",self.MEE_TS_std[i]," MSE TS/TR ",self.MSE_TS[i]/self.MSE_TR[i])

    def dynamic_learn_step(self,learning_rate,step, scaling,max_prove = 20, v_iniziale = np.inf, patience =0):
        """
        Perform dynamic learning step.

        Parameters:
        learning_rate (float): The current learning rate.
        step (int): The step size for each epoch.
        scaling (float): Scaling factor for adjusting the learning rate.
        max_prove (int): Maximum number of attempts to adjust the learning rate.
        v_iniziale (float): Initial validation error.
        patience (int): Number of consecutive epochs to wait for improvement before stopping.

        Returns:
        float: The adjusted learning rate.
        """
        nn = self.Neural_Network
        backup_layers = copy.deepcopy( nn.layers ) 
        prove = 0   # Counter for the number of attempts to adjust the learning rate
        lunghezza_lista = len(self.MSE_TR) # Length of the error lists

        # Continue adjusting the learning rate until the maximum number of attempts is reached
        while self.Early_stopping(learning_rate, step = 1, epoche = step, patience = patience, v_iniziale = v_iniziale) and prove < max_prove:
            # Adjust the learning rate, increase the attempt counter and restore the original neural netwtork layers
            learning_rate = learning_rate*scaling
            prove += 1
            nn.layers = copy.deepcopy( backup_layers)  
            
            # Remove errors from the error lists that were added during the previous attempt and update the length of the error lists
            temp=len(self.MSE_TR)
            [errore.pop() for errore in [self.MSE_TR, self.MSE_TS, self.MEE_TR,self.MEE_TS,self.MEE_TS_std,self.MSE_VL,self.MEE_VL,self.MEE_VL_std] for _ in range(temp - lunghezza_lista)]
            lunghezza_lista = len(self.MSE_TR)
        
        # Return the adjusted learning rate
        if prove < max_prove:
            return learning_rate
        return 0        # Return 0 if the maximum number of attempts is reached

    def dynamic_learn(self,step, epoche, max_prove=20,scaling = 0.5, scaling_up = False, verbouse = True, step_scaling_up = 5, patience = 0, learning_rate_init = 1):
        """
        Train the neural network using dynamic learning rate.

        Parameters:
        step (int): The step size for checking early stopping condition.
        epochs (int): Number of training epochs.
        max_prove (int): Maximum number of tries to adjust the learning rate before stopping.
        scaling (float): The factor to reduce the learning rate when adjusting.
        scaling_up (float): The factor to increase the learning rate when it's not improving. Don't use values = 1/scaling
        verbose (bool): If True, print training progress.
        step_scaling_up (int): The number of consecutive steps without improvement before increasing the learning rate.
        patience (int): The number of consecutive epochs to wait for improvement before stopping.
        learning_rate_init (float): Initial learning rate.

        Returns:
        function: A function that returns the adjusted learning rate based on the epoch number.
        """
        lunghezza_epoca = 1
        if self.Neural_Network.batch_size != -1:
            lunghezza_epoca = len(self.Neural_Network.batch.X_list)
        
        # Initialize learning rate and other variables
        learning_rate = learning_rate_init
        lista = [] # Store the learning rates
        lunghezza_lista_score = 0
        v_iniziale= np.inf
        
        for i in range((lunghezza_epoca*epoche)//step):
            learning_rate = self.dynamic_learn_step(learning_rate,step, scaling,max_prove, v_iniziale = v_iniziale, patience = patience)
            v_iniziale = self.MEE_VL[-1]    # Update v_iniziale to the last validation error
            
            if verbouse:
                for _ in range(lunghezza_lista_score,len(self.MSE_TR)):   print(_//lunghezza_epoca,"Epoca    TR/VL/TS MEE ", self.MEE_TR[_],"/", self.MEE_VL[_] ,"/",self.MEE_TS[_]," +- ",self.MEE_VL_std[_],"/",self.MEE_TS_std[_]," MSE TS/TR ",self.MSE_TS[_]/self.MSE_TR[_])
            
            lunghezza_lista_score = len(self.MSE_TR)
            lista.append(learning_rate)
            
            if learning_rate == 0:
                break
            
            if scaling_up and all(lista[-1] == _ for _ in lista[-step_scaling_up:]):
                 # If scaling_up is enabled and the learning rate hasn't changed for step_scaling_up steps, increase the learning rate
                learning_rate = learning_rate*scaling_up
        
        # Return a function that returns the adjusted learning rate based on the epoch number
        return lambda x: lista[x//step] if x//step < len(lista) else lista[-1]
    
    def Early_stopping(self, learning_rate, step, epoche, patience, backup = False, v_iniziale = np.inf):
        """
        Check for early stopping condition based on the validation error.

        Parameters:
        learning_rate (float): The current learning rate.
        step (int): The step size for each epoch.
        epochs (int): Total number of epochs to train.
        patience (int): Number of consecutive epochs to wait for improvement before stopping.
        backup (bool): If True, make a backup of the neural network layers when there's an improvement.
        v_iniziale (float): The initial validation error.

        Returns:
        bool: True if the early stopping condition is met (no improvement for `patience` epochs), False otherwise.
        """
        nn = self.Neural_Network
        p=0 # Counter for consecutive epochs without improvemen
        ep = 0 # Current epoch
        best_epoch = 0 # Epoch with the best validation error
        v = v_iniziale # Current validation error
        
        # Continue training until patience is reached or the maximum number of epochs is reached
        while ( p <= patience and ep < epoche):
            # Perform training steps for the current epoch
            for i in range(step) :  nn.step_train(learning_rate)
            
            ep += step
            
            # Calculate errors for training, validation, and testing datasets
            self.MSE_TR.append(mean_squared_error(self.Y_TR, nn.forward(self.X_TR)))
            self.MSE_TS.append( mean_squared_error(self.Y_TS, nn.forward(self.X_TS)))
            self.MSE_VL.append( mean_squared_error(self.Y_VL, nn.forward(self.X_VL)))
            
            self.MEE_TR.append(MEE(self.Y_TR, nn.forward(self.X_TR)))
            self.MEE_VL.append(MEE(self.Y_VL, nn.forward(self.X_VL)))
            self.MEE_TS.append( MEE(self.Y_TS, nn.forward(self.X_TS)) )
            
            # Calculate standard deviations for validation and testing MEE
            self.MEE_TS_std.append( MEE_std(self.Y_TS, nn.forward(self.X_TS)) )
            self.MEE_VL_std.append(MEE_std(self.Y_VL, nn.forward(self.X_VL)))
            
            # Check if there's no improvement in validation error
            if self.MEE_VL[-1] > v:     
                p+=1    # Increment patience counter
            else :
                # If there's an improvement, update variables
                if backup:
                    # Make a backup of the neural network layers
                    self.backup_layers = copy.deepcopy(nn.layers)   
                p=0     # Reset patience counter
                v = self.MEE_VL[-1]     # Update validation error
                self.best_epoch = ep    # Update the epoch with the best validation error
        
        # Return True if patience is reached (no improvement), False otherwise
        return p > patience
            
    def __init__(self,Neural_Network, X_TR, Y_TR, X_VL, Y_VL, X_TS = None, Y_TS = None):
        """
        Initialize the Train object.

        Parameters:
        Neural_Network (object): An initialized neural network object.
        X_TR (array-like): Input features for the training dataset.
        Y_TR (array-like): Target values for the training dataset.
        X_VL (array-like): Input features for the validation dataset.
        Y_VL (array-like): Target values for the validation dataset.
        X_TS (array-like, optional): Input features for the testing dataset. Default is None.
        Y_TS (array-like, optional): Target values for the testing dataset. Default is None.

        """
        # Store the provided neural network object
        self.Neural_Network = Neural_Network
        
         # Initialize lists to store MSE and MEE values for training, testing, and validation datasets
        self.MSE_TR  = []
        self.MSE_TS = []
        self.MSE_VL =[]
        self.MEE_TR = [] 
        self.MEE_TS = []
        self.MEE_TS_std = []
        self.MEE_VL_std =[]
        self.MEE_VL = []
        
        
        # Store the input features and target values for training, testing, and validation datasets
        self.X_TR = X_TR
        self.X_TS = X_TS if type(X_TS) != type(None) else X_VL
        self.Y_TR = Y_TR
        self.Y_TS = Y_TS if type(Y_TS) != type(None) else Y_VL
        self.X_VL = X_VL
        self.Y_VL = Y_VL

    def grafico_errore(self):
        fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

        axs[0].plot(self.MSE_TR, label=" Train MSE", linestyle=':')
        axs[0].plot(self.MSE_TS, label=" Test MSE")
        axs[0].set_title("MSE")
        axs[0].set_xlabel("epoch")

        x = np.array(range(len(self.MEE_TR)))
        axs[1].plot(self.MEE_TR, label=" Train MEE", linestyle=':')
        axs[1].plot(self.MEE_TS, label="Test MEE")
        axs[1].set_title("MAE")
        axs[1].set_xlabel("epoch")

        axs[0].legend(loc='best', fontsize=7)
        axs[1].legend(loc='best', fontsize=7)
        
        return fig,axs
    
if __name__=="__main__":
    import pandas as pd  
    from sklearn.model_selection import train_test_split
    import time
    df=pd.read_csv("ML-CUP23-TR.csv",comment="#",header = None)
    y=df.iloc[:,-3:]
    x=df.iloc[:,1:-3]
    
    def TR_VL_TS(x,y, bootstrap = False, size_bootstrap = 1,size_validation= 0.111):
        x = x.values
        y = y.values 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1111, random_state=0)
         
        if bootstrap:
            indici = np.random.randint(0,x_train.shape[0],int(size_bootstrap*x_train.shape[0]))

            x_val = x_train[~np.isin(np.arange(x_train.shape[0]),indici)]
            y_val = y_train[~np.isin(np.arange(y_train.shape[0]),indici)]

            x_train = x_train[indici]
            y_train = y_train[indici]
        else :
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=size_validation,random_state=None)
    

        from sklearn.preprocessing import PolynomialFeatures
        from sklearn import preprocessing
        #-----------------------------------------
        #---- DATA PREPROCESSING AND SPLIT TRAINING TEST
        poly = PolynomialFeatures(2)
        X_TR_poly = poly.fit_transform(x_train)
        X_TS_poly = poly.transform(x_test)
        X_VL_poly = poly.transform(x_val)

        X_TR=np.arctanh(X_TR_poly[:,1:])
        X_TS=np.arctanh(X_TS_poly[:,1:])
        X_VL=np.arctanh(X_VL_poly[:,1:])
        
        return X_TR, y_train, X_TS, y_test, X_VL, y_val
    #---------------------------------------------------
    X_TR, Y_TR, X_TS, Y_TS, X_VL, Y_VL = TR_VL_TS(x,y, bootstrap = False,size_validation=0.1)
    tin = time.time()
    alfa=0.36
    lamda = 0.00055
    batch_size=-1
    epoche=100000

    nn= (NN.Neural_Network(3,65))
    nn.add_layer(150,activation= "tanh")
    
    



    nn.set_train(X_TR,Y_TR,batch_size=batch_size,Loss_Function = "MSE",random_state= None )
    nn.gradient_treshold = 10
    nn.Inizialization(type="Xavier_uniform",scale=1)
    nn.lamda_2=lamda
    nn.alfa=alfa


    train=Train(nn, X_TR, Y_TR, X_VL, Y_VL, X_TS, Y_TS)

    learning_rate =train.dynamic_learn(step = 10, epoche = epoche, max_prove=200,scaling = 0.9, verbouse = True, patience = 0)
    # plt.plot(list(map(learning_rate,range(epoche))))
    
    

    train.grafico_errore()
    print("esecuzione di ", time.time()-tin, " s")
    plt.show()
    
