import numpy as np

class BaseLayer:
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.h -= eta * self.grad_b

class MiddleLayer(BaseLayer):
    def __init__(self, n_upper, n):
        # He 초깃값
        self.w = np.random.randn(n_upper, n) * np.sqrt(2/n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1) # ReLu 미분

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.x.T)