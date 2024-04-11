import numpy as np

class loss_function():
    def __init__(self, function):
        self.loss_function = function
        
    def loss_out(self, y, w, b):
        if self.loss_function == "mse":
            loss = 2 * np.sum((w * x - y) ** 2) / len(y) # loss of model 
            
            w_loss_prime = 2 * np.sum((w * x - (y - b)) * x) / len(y) #differentiation w
            b_loss_prime = 2 * np.sum((w * x - (y - b))) / len(y) # differentiation b
            
        else:
            raise "Choose proper loss function"
            
        return loss, w_loss_prime, b_loss_prime
