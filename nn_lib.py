import numpy as np

class Layer:
    def __init__(self, from_, to, activation):
        self.W = np.random.normal(scale=np.sqrt(2 / (from_ + to)), size=(from_, to))
        self.b = np.zeros(to)
        self.activation = activation
        
    def __call__(self, X):
        return self.activation(X @ self.W + self.b)

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

    def backpropagate(self, xn, yn):
        # forward pass
        x = [xn]
        s = []
        for l in range(self.L):
            s.append(x[l] @ self.layers[l].W + self.layers[l].b)
            x.append(self.layers[l].activation(s[-1]))
        
        # backward pass
        dl_dx = self.loss.derivative(x.pop(), yn)
        dl_ds = dl_dx * self.layers[self.L - 1].activation.derivative(s.pop())
        dl_dw = [x.pop().T @ dl_ds]
        dl_db = [dl_ds.sum(axis=0)]
        for l in reversed(range(self.L - 1)):
            dl_dx = dl_ds @ self.layers[l + 1].W.T
            dl_ds = dl_dx * self.layers[l].activation.derivative(s.pop())
            dl_dw.insert(0, x.pop().T @ dl_ds)
            dl_db.insert(0, dl_ds.sum(axis=0))
            
        return dl_dw, dl_db
    
    def train(self, X, y, epochs=10, batch_size=128, gamma=0.01):
        for _ in range(epochs):
            for b in range(0, X.shape[0], batch_size):
                dl_dw, dl_db = self.backpropagate(X[b: b + batch_size], y[b: b + batch_size])
                self.SGD(dl_dw, dl_db, gamma)
                
    def SGD(self, dl_dw, dl_db, gamma):
        for l in range(self.L):
            self.layers[l].W -= gamma * dl_dw[l]
            self.layers[l].b -= gamma * dl_db[l]
