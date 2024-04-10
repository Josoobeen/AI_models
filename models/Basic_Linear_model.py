import numpy as np


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
            
class Optimizer():
    def __init__(self, optimizer, learning_rate = 0.01):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        
    def optimizer_out(self, w_loss_prime, b_loss_prime, w, b):
        if self.optimizer == "gradient_descent":

            gradient_w = w_loss_prime
            gradient_b = b_loss_prime
            
            w -= self.learning_rate * gradient_w
            b -= self.learning_rate * gradient_b
                
        else:
            raise "Choose proper optimizer"
            
        return w, b



dense = Linear(0.01, 0.01)
x = np.arange(1, 100) # training data
y = 3 * x + 1    # training data

for i in range(5000):
    dense.train(x, y)
print(dense.show_weights(), dense.show_bias())

for i in range(5000):
    dense.train(x, y)
print(dense.show_weights(), dense.show_bias())
