import numpy as np

# Define the framework functions 
class Dense_layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.rand(n_inputs, n_outputs)
        self.biases = np.random.rand(n_outputs)
    def forward(self, x):
        self.outputs = np.dot(x, self.weights) + self.biases

class Activation:
    def forward(self, inputs):
        self.outputs = np.maximum(x,0) 

class Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs)
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
class Sigmoid:
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))

class categorical_crossEntropy:
    def __init__(self, y_pred, y):
        a = -np.log(y_pred)
        b = y.T
        return a*b


if __name__ == "__main__":
    x = np.array([  [1, 2, 3],
                    [2, 4, 1],
                    [3, 1, 2],
                    [4, 5, 3]  ])

    L1 = Dense_layer(3, 4)
    A = Softmax()

    L1.forward(x)
    A.forward(L1.outputs)
    print(np.sum(A.outputs, axis=1))