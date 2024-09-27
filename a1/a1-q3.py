import math
import matplotlib.pyplot as plt
import numpy as np

DURATION = 30 #s
SAMPLING_T = 0.1 #s

PROFILE = 3

def x_delta(theta, time):
    if PROFILE == 0:
        total_dis = 1 * SAMPLING_T
        return total_dis * math.cos(theta)
    
    if PROFILE == 1:
        total_dis = 0 * SAMPLING_T
        return total_dis * math.cos(theta)
    
    if PROFILE == 2:
        total_dis = 1 * SAMPLING_T
        return total_dis * math.cos(theta)
    
    if PROFILE == 3:
        total_dis = (1 + 0.1 * math.sin(time)) * SAMPLING_T
        return total_dis * math.cos(theta)

def y_delta(theta, time):
    if PROFILE == 0:
        total_dis = 1 * SAMPLING_T
        return total_dis * math.cos(theta)
    
    if PROFILE == 1:
        total_dis = 0 * SAMPLING_T
        return total_dis * math.cos(theta)
    
    if PROFILE == 2:
        total_dis = 1 * SAMPLING_T
        return total_dis * math.sin(theta)
    
    if PROFILE == 3:
        total_dis = (1 + 0.1 * math.sin(time)) * SAMPLING_T
        return total_dis * math.sin(theta)

def theta_delta(time):
    if PROFILE == 0:
        return 0 * SAMPLING_T

    if PROFILE == 1:
        return 0.3 * SAMPLING_T

    if PROFILE == 2:
        return 0.3 * SAMPLING_T

    if PROFILE == 3:
        return (0.2 + 0.5 * math.cos(time)) * SAMPLING_T



def main():
    theta = 0
    cur_x = 0   
    cur_y = 0
    data = []

    for time in np.arange(0, DURATION, SAMPLING_T):
        cur_x += x_delta(theta, time)
        cur_y += y_delta(theta, time)
        theta += theta_delta(time)

        data.append((cur_x, cur_y, theta, time))

        time += 0.1

    x_vals, y_vals, theta_vals, time_vals = zip(*data)

    print(time_vals)

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))  # Two subplots, stacked vertically

    # First subplot: x vs. y
    ax1.plot(x_vals, y_vals, 'bo-', label='x vs. y')

    # Customize the first plot
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('X vs Y')
    ax1.legend()
    ax1.grid(True)

    # Second subplot: x, y, and theta vs. time
    ax2.plot(time_vals, x_vals, 'bo-', label='x vs. time')  # x vs time
    ax2.plot(time_vals, y_vals, 'ro-', label='y vs. time')  # y vs time
    ax2.plot(time_vals, theta_vals, 'go-', label='theta vs. time')  # theta vs time

    # Customize the second plot
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X, Y, Theta')
    ax2.set_title('X, Y, and Theta vs Time')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout for better spacing between plots
    plt.tight_layout()

    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()