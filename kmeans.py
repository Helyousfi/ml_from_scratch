import numpy as np
import matplotlib.pyplot as plt

# Create the data
n = 500 # samples
p = 2   # features
k = 3   # clusters

X = np.random.random((n,p))

#
# initilize the centers
#

centers = X[np.random.choice(n, k, replace=False)]
closest = np.zeros(n).astype(int)

while True:
    prev_closest = closest.copy()
    
    #
    # Calculate the distance
    #
    distances = np.zeros((n, k))
    for i in range(k):
        distances[:, i] = ((X - centers[i])**2).sum(axis=1)**0.5

    #
    # Assign clusters
    #
    closest = np.argmin(distances, axis = 1)

    #
    # Update centers
    #  
    for i in range(k):
        centers[i] = np.mean(X[closest == i], axis=0)
    
    if all(prev_closest == closest):
        break
    
plt.scatter(X[:,0], X[:,1], c=closest)
plt.title("k means clustering")
plt.show()

