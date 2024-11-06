from scipy.linalg import solve
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd

def plot_ellipse(Q, b):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    S = scipy.linalg.sqrtm(np.linalg.inv(Q))
    
    eigvals, eigvecs = np.linalg.eigh(S)
    aa, bb = eigvals

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_param = np.array([aa * unit_circle[0], bb * unit_circle[1]])
    ellipse_points = eigvecs @ ellipse_param + b[:, np.newaxis]

    ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', label='Fitted Ellipse')
    ax.plot(b[0], b[1], 'ro', label='Center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True)

x1, y1 = (2.92, -6.01)
x2, y2 = (3.40, -7.20)
x3, y3 = (4.99, -7.84)
x4, y4 = (5.48, -7.04)
x5, y5 = (4.20, -5.91)

A_prime = np.array([[x1**2, 2*x1*y1, y1**2, -2*x1, -2*y1],
              [x2**2, 2*x2*y2, y2**2, -2*x2, -2*y2],
              [x3**2, 2*x3*y3, y3**2, -2*x3, -2*y3],
              [x4**2, 2*x4*y4, y4**2, -2*x4, -2*y4],
              [x5**2, 2*x5*y5, y5**2, -2*x5, -2*y5]])
b_prime = np.array([-1,-1,-1,-1,-1])

x = solve(A_prime, b_prime)
print(x)

A = x[0]
B = x[1]
C = x[2]
D = x[3]
E = x[4]

Z = np.array([D, E])
Y = np.array([[A, B],
              [B, C]])
V = np.array([[D], [E]])
S = np.array([1])


alpha = 1.0 / ((Z @ np.linalg.inv(Y) @ V) - S)


Q = alpha * Y
b = (np.linalg.inv(Y) @ V)

print(f"Q: {Q}")
print(f"b: {b}")

plot_ellipse(Q, b)
plt.show()
