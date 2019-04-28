import numpy as np

# identity
class Identity:
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return np.ones(x.shape)

# sigmoid/logistic function
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        exp_ = np.exp(-x)
        return exp_ / (1 + exp_) ** 2

# rectified linear unit
class Relu:
    def __call__(self, x):
        return (x > 0) * x

    def derivative(self, x):
        return np.sign(x)

# leaky relu
class LRelu:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return lambda x: (x > 0) * x + (x < 0) * a * x

    def derivative(self, x):
        return lambda x: (x > 0) * 1 + (x < 0) * a

# softplus/smoothrelu
class Softplus:
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return sigmoid(x)

# hyperbolic tangent
class Tanh:
    def __call__(self, x):
        exp_ = np.exp(2 * x)
        return (exp_ - 1) / (exp_ + 1)

    def derivative(self, x):
        exp_ = np.exp(2 * x)
        return 4 * exp_ / (exp_ + 1) ** 2 
