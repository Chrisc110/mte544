"""
Student Name & Last Name: 
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q2_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd
def plot_ellipse(Q, b, ax):
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

def visualize_data(p, ax, inliers, threshold):
    ax.scatter(p[:, 0], p[:, 1], color='red', alpha=0.5, label='Raw Measurements (Ellipse)')
    ax.scatter(inliers[:, 0], inliers[:, 1], color='purple', alpha=0.7, label='Inliers')

    for point in inliers:
        circle = plt.Circle(point, threshold, color='orange', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)


def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def fit_ellipse_subset(points):
    # Given some number of points (you have to determined this), 
    # construct an ellipse that fits through those points.
    

    ##### ADD your code here : #####

    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]
    x5, y5 = points[4]

    A_prime = np.array([[x1**2, 2*x1*y1, y1**2, -2*x1, -2*y1],
                        [x2**2, 2*x2*y2, y2**2, -2*x2, -2*y2],
                        [x3**2, 2*x3*y3, y3**2, -2*x3, -2*y3],
                        [x4**2, 2*x4*y4, y4**2, -2*x4, -2*y4],
                        [x5**2, 2*x5*y5, y5**2, -2*x5, -2*y5]])
    b_prime = np.array([-1,-1,-1,-1,-1])

    x = scipy.linalg.solve(A_prime, b_prime)
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
    b = np.linalg.inv(Y) @ V
    ##### END #####
    return Q, b

def ransac_ellipse(data, num_iterations=1000, threshold=0.2):
    inliers = []
    # Given the data sets, perform RANSAC to find the best Q and b as well as the inliers
    # Hint: You should use fit_ellipse_subset 
    # Hint: in some case, the Q matrix might not be positive defintie, use is_positive_definite to check.

    ##### ADD your code here : #####

    # detect if the ellipse is too eccentric*
    [e_vals,_] = np.linalg.eigh(scipy.linalg.sqrtm(np.linalg.inv(Q)))
    if e_vals[0]>20 or e_vals[1]>20:
        continue
    ...
    ##### END #####
    return Q,b,inliers

if __name__ == "__main__":
    # Load the data from CSV file and select N random points
    N = 500
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    # dataset is p
    
    Q, b_est, inliers = ransac_ellipse(dataset)
    
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.5)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    plt.show()