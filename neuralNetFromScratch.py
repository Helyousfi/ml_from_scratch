import numpy as np

# Define the framework functions 
class Dense_layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.rand(n_inputs, n_outputs)
        self.biases = np.random.rand(n_outputs)
    def forward(self, x):
        self.outputs = np.dot(x, self.weights) + self.biases

class Activation:
    # Leaky ReLU activation function
    def forward(self, inputs, alpha = 0):
        self.outputs = np.maximum(inputs, alpha * inputs) 

class Softmax:
    # The sum over the axis 1 must be 1
    def forward(self, inputs):
        exp_vals = np.exp(inputs)
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
class Sigmoid:
    # for logistic regression
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))

class categorical_crossEntropy:
    def __init__(self, y_pred, y):
        self.y_pred = y_pred / np.max(y_pred)
        self.y_pred = np.clip(self.y_pred, 1e-7, 1-1e-7)
        self.outputs = np.mean(-np.dot(self.to_categorical(y), np.log(self.y_pred).T)\
            - np.dot((1 - self.to_categorical(y)), np.log(1 - self.y_pred).T) )
    
    # Convert y to categorical
    def to_categorical(self, y):
        y_cat = np.zeros( (len(y), np.max(y)+1) )
        for k in range(len(y)):
            y_cat[k][y[k]] = 1
        return y_cat


if __name__ == "__main__":
    # Classification problem
    x = np.array([  [1, 2, 3],
                    [2, 4, 1]  ])
    y = np.array( [0, 2] )

    # Create the NN
    L1 = Dense_layer(3, 3)
    A1 = Softmax()
    
    # Forward pass
    L1.forward(x)
    A1.forward(L1.outputs)
    
    Loss = categorical_crossEntropy(L1.outputs, y)
    print(Loss.outputs)