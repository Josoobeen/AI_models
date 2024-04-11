import numpy as np
from Loss_function import loss_function
from Optimizer import Optimizer

class Linear():
    def __init__(self, a, b):
        self.weights = a
        self.bias = b

    def train(self, x, y, loss_f = "mse", optimizer = "gradient_descent"):
        
        y_ = self.weights * x + self.bias
        
        loss = loss_function(loss_f)
        loss_, w_loss_prime, b_loss_prime = loss.loss_out(y, self.weights, self.bias)
        
        
        optimizer = Optimizer(optimizer)
        self.weights, self.bias = optimizer.optimizer_out(w_loss_prime, b_loss_prime, self.weights, self.bias)
            
            
        return loss_, self.weights, self.bias
        
    
    def predict(self, x):
        return self.weights * x + self.bias
    
    def show_weights(self):
        return self.weights
    
    def show_bias(self):
        return self.bias



dense = Linear(0.01, 0.01)
x = np.arange(1, 100) # training data
y = 3 * x + 1    # training data

for i in range(5000):
    dense.train(x, y)
print(dense.show_weights(), dense.show_bias())

for i in range(5000):
    dense.train(x, y)
print(dense.show_weights(), dense.show_bias())
