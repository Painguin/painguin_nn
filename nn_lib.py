import numpy as np

class Layer:
    def __init__(self, from_, to, activation):
        self.W = np.random.normal(scale=np.sqrt(2 / (from_ + to)), size=(from_, to))
        self.b = np.zeros(to)
        self.activation = activation
        
    def __call__(self, X):
        return self.activation(X @ self.W + self.b)
    
    def forward(self, x):
        s = x @ self.W + self.b
        y = self.activation(s)
        return y, s
    
    def backward(self, x, s, dl_dx):
        dl_ds = dl_dx * self.activation.derivative(s)
        dl_dw = x.T @ dl_ds
        dl_db = dl_ds.sum(axis=0)
        dl_dx_prev = dl_ds @ self.W.T
        
        return dl_dw, dl_db, dl_dx_prev

class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers = layers
        self.L = len(layers)
        self.loss = loss
        
    def __call__(self, X):
        x = X
        for l in range(self.L):
            x = self.layers[l](x)
        return x
    
    def forward(self, x):
        x_arr = [x]
        s_arr = []
        
        for layer in self.layers:
            x_lay, s_lay = layer.forward(x_arr[-1])
            x_arr.append(x_lay)
            s_arr.append(s_lay)
            
        return x_arr, s_arr
    
    def backward(self, x_arr, s_arr, dl_dx):
        dl_dw = []
        dl_db = []
        
        for layer, x, s in zip(reversed(self.layers), reversed(x_arr), reversed(s_arr)):
            dl_dw_l, dl_db_l, dl_dx = layer.backward(x, s, dl_dx)
            dl_dw.insert(0, dl_dw_l)
            dl_db.insert(0, dl_db_l)
            
        return dl_dw, dl_db

    def backpropagate(self, x, y):
        # forward pass
        x_arr, s_arr = self.forward(x)
        
        # backward pass
        dl_dx = self.loss.derivative(x_arr.pop(), y)
        dl_dw, dl_db = self.backward(x_arr, s_arr, dl_dx)
        
        return dl_dw, dl_db
    
    def fit(self, X, y, epochs=10, batch_size=128, gamma=0.01):
        for _ in range(epochs):
            for b in range(0, X.shape[0], batch_size):
                dl_dw, dl_db = self.backpropagate(X[b: b + batch_size], y[b: b + batch_size])
                for layer, dl_dw_l, dl_db_l in zip(self.layers, dl_dw, dl_db):
                    layer.W -= gamma * dl_dw_l
                    layer.b -= gamma * dl_db_l