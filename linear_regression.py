import numpy as np
import matplotlib.pyplot as plt

"""
h(x) = thetaT*x
l(theta) = 1/2m * sum_over_m(thetaT * x- y)²
"""


X = np.array([1, 2, 4, 5, 6 ,8])
y = np.array([2.5, 4.5, 7, 10.4, 13, 15])

theta = np.array([4, 2])

print(np.matmul(theta.T, np.array([[1, 1, 1, 1, 1, 1], X])))


def hypothesis(theta, X):
    # h(x) = theta.T * x
    X_new = np.array([[1, 1, 1, 1, 1, 1], X])
    return np.matmul(theta, X_new)

def loss(y_pred, y):
    return np.sum((y_pred - y)**2)

alpha = 0.01

m = 1
# batch gradient descent 
y_pred = hypothesis(theta, X)
l = loss(y_pred, y)
for j in range(1000):
    y_pred = hypothesis(theta, X)
    theta = theta - alpha * np.matmul((hypothesis(theta, X) - y), np.array([[1, 1, 1, 1, 1, 1], X]).T)
    print(theta)

plt.plot(np.arange(1, 10),hypothesis(theta, np.arange(1, 10)))
plt.show()








"""
plt.scatter(X, y)
plt.show()


def loss(y_pred, y):
    return sum((y-y_pred)**2)

def forward(X, theta0, theta1):
    return [theta1*x + theta0 for x in X]

def grad(l):
    return 

def gradient_descent(alpha, theta, X, y):
    y_pred = forward(X, theta)
    l = loss(y_pred, y)
    theta = theta - alpha * 1/m * (x -   

print(forward(X, theta))
"""