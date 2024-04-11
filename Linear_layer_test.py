from model.Basic_Linear_model import Linear
import numpy as np

data_amount = 1000
dense = Linear(0.01, 10)
x = np.arange(1, data_amount)/data_amount # training data
y = 3 * x + 1    # training data



for i in range(5000):
    dense.train(x, y)
print(dense.show_weights(), dense.show_bias())
print(dense.predict(x), "\n", y)
