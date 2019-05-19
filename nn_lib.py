import numpy as np

class Layer:
    def forward(self, x):
        return NotImplementedError()
    
    def backward(self, x):
        return NotImplementedError()

    def __call__(self, x):
        return self.forward(x)
    
    def params(self):
        return []


class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dl_dy):
        for layer in reversed(self.layers):
            dl_dy = layer.backward(dl_dy)
        return dl_dy

    def params(self):
        return [param for layer in self.layers for param in layer.params()]


class Linear(Layer):
    def __init__(self, from_, to):
        self.w = np.random.normal(scale=np.sqrt(2 / (from_ + to)), size=(from_, to))
        self.b = np.zeros(to)
        self.dl_dw = np.zeros_like(self.w)
        self.dl_db = np.zeros_like(self.b)
        
    def forward(self, x):
        self.input = x
        return x @ self.w + self.b
    
    def backward(self, dl_dy):
        self.dl_dw += self.input.T @ dl_dy
        self.dl_db += dl_dy.sum(axis=0)
        return dl_dy @ self.w.T
    
    def params(self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]
    
class ReLU(Layer):
    def forward(self, x):
        self.input = x
        return (x > 0) * x

    def backward(self, dl_dy):
        return np.sign(self.input) * dl_dy

class Tanh(Layer):
    def forward(self, x):
        self.input = x
        exp_ = np.exp(2 * x)
        return (exp_ - 1) / (exp_ + 1)

    def backward(self, dl_dy):
        exp_ = np.exp(2 * self.input)
        return dl_dy * 4 * exp_ / (exp_ + 1) ** 2

class Loss:
    def forward(self, pred, target):
        return NotImplementedError()

    def backward(self, pred, target):
        return NotImplementedError()
    
    def __call__(self, pred, target):
        return self.forward(pred, target)

class MSE(Loss):
    def forward(self, pred, target):
        return np.sum(
            (target - pred) ** 2
        )

    def backward(self, pred, target):
        return -2 * (target - pred)

class SGD:
    def __init__(self, params, gamma):
        self.params = params
        self.gamma = gamma

    def update(self):
        for param, dl_param in self.params:
            param -= self.gamma * dl_param
            dl_param.fill(0)

class Model:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
    
    def fit(self, x, y, epochs=10, batch_size=128):
        for _ in range(epochs):
            for b in range(0, x.shape[0], batch_size):
                output = self.model(x[b: b + batch_size])
                dl_dy = self.loss.backward(output, y[b: b + batch_size])
                self.model.backward(dl_dy)
                self.optimizer.update()