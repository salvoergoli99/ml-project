import numpy as np
class LossFunction:
    """
    Definition of various loss Functions

    :type: name of the loss fuctione. Could be "MSE", "HUBER" etc
    :d : desired output
    :delta: threshold for Huber loss, ignore for MSE
    :return: Loss fuctin class. Can be used by self.function and self. gradient to calculate loss and gradient
    """
    def __init__(self,type,d,delta=1):
        match type:
            case "MSE":
                self.type=type
                self.function=     lambda y : np.mean((y-d)**2,axis=1)/2 
                self.gradient=     lambda y : y-d  
            case "HUBER":
                def huber_loss(y_pred, y, delta):
                    d = np.abs(y - y_pred)
                    return np.mean(np.where(d <= delta, 0.5*d**2, delta * (d - 0.5 * delta)) , axis=1)
                
                def huber_loss_gradient(y_pred, y, delta):
                    d = np.abs(y - y_pred)
                    return np.where(d <= delta, y_pred-y, delta * np.sign(y_pred-y)) 
                
                self.type=type
                self.function= lambda t: huber_loss(t,d,delta)
                self.gradient=     lambda t: huber_loss_gradient(t,d,delta)
