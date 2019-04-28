import numpy as np

class MSE:
    def __call__(self, pred, target):
        return np.sum((pred - target) ** 2)
    
    def derivative(self, pred, target):
        return -2 * (target - pred)
