# AI_models
I build AI models


## Linear Layer
Basic Linear model.
In this model, paper assumed data has linear data w(weight) * x + b(Bias).
first when model build, random weights and bias is made.


#### in test_training
I use Mean Squared Error, witch means, loss is (y - pred_y) ^ 2.
I use Gradient descent, with means, differentiation weights, bias.

w calculate => (w*x - (y - b))^2/dw => w_loss_prime = 2 * sum((w * x - (y - b)) * x) / n

b calculate => (w*x - (y - b))^2/db => b_loss_prime = 2 * sum((w * x - (y - b))) / n

Because I use Gradient descent, if x cannot be bigger for train. x is better to be under 1.
