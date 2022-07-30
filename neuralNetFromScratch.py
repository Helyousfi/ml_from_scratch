import numpy as np

# Implement a dense layer from scratch
class Dense_layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.rand(n_inputs, n_outputs)
        self.biases = np.random.rand(n_outputs)
    def forward(self, x):
        self.outputs = np.dot(x, self.weights) + self.biases

# Implement the activation functions
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

# The loss function for classification
class Categorical_crossEntropy:
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

# Conv2d layer
class Conv2D:
    def __init__(self, in_channels, out_channels, 
                input_shape, filter_shape, 
                stride, padding = "same"):
        """
        in_channels : the number of the input image channels
        out_channels : the number of the output channels
        input_shape : (W, H) the shape of the image
        filter_shape : the size of the filter 
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self.filter_shape = filter_shape 
        self.stride = stride
        self.padding = padding
    
    def convolution(self, image, convFiler):
        if self.padding == "same":
            out_shape = image.shape
            new_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2)) 
            new_image[1:new_image.shape[0]-1, 1:new_image.shape[1]-1] = image
            image = new_image
        elif self.padding == "valid": 
            out_shape = (image.shape[0] - 2, image.shape[1] - 2) 
        image_out = np.zeros(out_shape)
        #TODO : add stride
        for y in range(out_shape[0]):
            for x in range(out_shape[1]):
                S = 0
                for n in range(-convFiler.shape[0]//2+1, convFiler.shape[0]//2 + 1):
                    for m in range(-convFiler.shape[1]//2+1, convFiler.shape[1]//2 + 1):
                        S += image[y + 1 + n, x + 1 + m] * \
                            convFiler[n + convFiler.shape[1]//2, m + convFiler.shape[0]//2]
                image_out[y, x] = S
        return image_out

    def forward(self, input):   # Forward pass to apply convolutions
        self.weights = np.random.rand(self.out_channels,  
                        self.in_channels, *self.filter_shape)
        self.biases = np.random.rand(self.out_channels, 
                        *self.filter_shape)
        self.output = np.zeros((self.out_channels, 
                        input.shape[0], input.shape[1]))
        
        for i in range(self.out_channels):
            s = np.zeros((input.shape[0], input.shape[1]))
            for k in range(self.in_channels):
                s += self.convolution(input[:, :, k], self.weights[i][k])
            self.output [i] = s
    
class PoolingLayer:
    def __init__(self, pool_size=(2, 2), stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, input):
        self.output = np.zeros((input.shape[0]//self.pool_size[0], 
                                input.shape[1]//self.pool_size[1],
                                input.shape[2]))
        for c in range(input.shape[2]):
            for i in range(0, input.shape[0]-self.stride, self.stride):
                for j in range(0, input.shape[1]-self.stride, self.stride): 
                    temp = []
                    for k1 in range(self.pool_size[0]):
                        for k2 in range(self.pool_size[1]):
                            temp.append(input[i+k1][j+k2][c])
                    self.output[i//self.stride][j//self.stride][c] = np.max(temp)
        
class Flatten:
    def __init__(self):
        pass
    def forward(self, input):
        self.output = []
        for batch in range(input.shape[0]):
            for c in range(input.shape[1]):
                for y in range(input.shape[2]):
                    for x in range(input.shape[3]):
                        self.output.append(input[batch][c][y][x])
        self.output = np.array(self.output)

                

        
if __name__ == "__main__":
    X = np.ones((1,5,5,3))
    Conv_layer = Flatten()
    Conv_layer.forward(X)
    print("output shape : ", Conv_layer.output.shape)



    """
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
    
    Loss = Categorical_crossEntropy(L1.outputs, y)
    print(Loss.outputs)
    """
