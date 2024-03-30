class Linear():
  def __init__(self, a, b):
    self.alpha = a
    self.beta = b

  def train(self, x, y):
    y_ = self.alpha * x + self.beta
    loss_alpha = y_ - y
    
  


