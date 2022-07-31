import numpy as np

# Implement a dense layer from scratch
class Dense_layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.rand(n_inputs, n_outputs)
        self.biases = np.random.rand(n_outputs)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases

    def gradient(self, y_grad):
        self.grad_weights = np.dot(self.inputs.T, y_grad)
        self.grad_biases = y_grad

# Implement the activation functions
class LeakyReLU:
    # Leaky ReLU activation function
    def __init__(self, alpha = 0):
        self.alpha = alpha 

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.where(self.inputs>0, self.inputs, self.alpha * self.inputs)

    def gradient(self):
        self.grad = np.where(self.inputs>0, 1, self.alpha)

# The mean squared error
class MSE:
    def forward(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred
        self.outputs = np.mean((y - y_pred) ** 2)
    
    def gradient(self):
        self.y_grad = 2 * (self.y - self.y_pred)



if __name__ == "__main__":
    ## Create the dataset 
    X = np.array([1, 2, 4, 5, 7, 6, 11, 14, 4])
    y = np.array([2, 3, 8, 11, 15, 13, 23, 27, 20])

    beta = 0.0002

    # Create the network
    ## first layer
    Layer1 = Dense_layer(1, 1)


    ## activation function
    relu = LeakyReLU(alpha=0)
    ## loss
    Mse = MSE()

    for i in range(50):
        for i in range(len(X)):
            # forward pass
            Layer1.forward(X[i])
            relu.forward(Layer1.outputs)
            Mse.forward(y[i], relu.outputs)
            L = Mse.outputs

            # update params
            Mse.gradient() 
            relu.gradient()
            Layer1.gradient(Mse.y_grad * relu.grad)
            

            Layer1.weights = Layer1.weights + beta * Layer1.grad_weights
            Layer1.biases = Layer1.biases + beta * Layer1.grad_biases

        print("The loss : ", L)

    inp = np.linspace(1, 20, 100)
    outp = []
    for x in inp:
        Layer1.forward(x)
        relu.forward(Layer1.outputs)
        a = relu.outputs
        outp.append(a[0][0])
    import matplotlib.pyplot as plt
    print(outp)
    plt.plot(inp, outp)
    plt.scatter(X, y)
    plt.show()

