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
