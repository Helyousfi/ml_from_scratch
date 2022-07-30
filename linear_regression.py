import numpy as np
import matplotlib.pyplot as plt

"""
h(x) = thetaT*x
l(theta) = 1/2m * sum_over_m(thetaT * x- y)²
"""

def transpose(L):
    n = len(L)
    m = len(L[0])
    L_transpose = []
    for j in range(m):
        temp = []
        for i in range(n):
            temp.append(L[i][j])
        L_transpose.append(temp)
    return L_transpose


def matrixMul(M, L):

    n1, m1 = len(M), len(M[0])
    n2, m2 = len(L), len(L[0])

    if m1 == n2:
        P = []
        for i in range(n1):
            line = []
            for j in range(m2):
                Cij = 0
                for k in range(m1):
                    Cij += M[i][k] * L[k][j]
                line.append(Cij)
            P.append(line)
        return P

def matrixMulalpha(alpha, L):
    n, m = len(L), len(L[0])
    for i in range(n):
        for j in range(m):
            L[i][j] = alpha * L[i][j]
    return L

def diff(a, b):
    n = len(a)
    temp = []
    for i in range(n):
        temp.append(a[i] - b[i])
    return [temp]

def diff2(a, b):
    n = len(a)
    m = len(a[0])
    d = []
    for i in range(n):
        temp = []
        for j in range(m):
            temp.append(a[i][j] - b[i][j])
        d.append(temp)
    return d


# Create the data 
X = [ [1, 1, 1, 1, 1, 1], 
      [1, 2.5, 4, 5, 6, 7.5]]
y = [2.5, 4.5, 7, 10.4, 13, 15]

theta = [ [4], 
          [2] ]

print(X)
print(matrixMul(transpose(theta), X)[0])



def hypothesis(theta, X):
    # h(x) = theta.T * x
    return matrixMul(transpose(theta), X)[0]

def loss(y_pred, y):
    n = len(y_pred)
    s = 0
    for i in range(n):
        s += (y_pred[i] - y[i])**2
    return s

alpha = 0.01

m = 1
# batch gradient descent 
y_pred = hypothesis(theta, X)
l = loss(y_pred, y)
for j in range(1000):
    y_pred = hypothesis(theta, X)
    print(diff(y_pred, y))
    theta = diff2(theta, transpose(matrixMulalpha(alpha, matrixMul(diff(hypothesis(theta, X), y), transpose(X)))))
    print(theta)








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
