"""
Student Name & Last Name: Christopher Chung
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q4_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""

import numpy as np

def decode_measurement(measurements):
    readings = []
    for measurement in measurements:
        readings.append(np.argmax(measurement) + 1)  # +1 for 1-based indexing
    return readings

def moodeng_behavior_update(state, A):
    # Given the current state, and the transition matrix A, 
    # randomly return the next state of Moo-Deng based on A
    ##### ADD your code here : #####

    # Convert the integer state into a matrix so we can multiply by A
    state_matrix = 0
    if state == 1:
        state_matrix = np.array([[1], [0], [0]])
    elif state == 2:
        state_matrix = np.array([[0], [1], [0]])
    else:
        state_matrix = np.array([[0], [0], [1]])

    next_state_matrix = A @ state_matrix

    # based on the probabilities, randomly pick a state
    next_state = np.random.choice([1, 2, 3], p=next_state_matrix.flatten())
    ##### END #####
    return next_state

def sensor_measurement(state, C):
    # Given a state, and the matrix C, 
    # randomly return the encoded measurement based on C
    # Note that : 
    # F -> np.array([1,0,0])
    # R -> np.array([0,1,0])
    # P -> np.array([0,0,1])
    ##### ADD your code here : #####
    
    # Convert the integer state into a matrix so we can multiply by C
    state_matrix = 0
    if state == 1:
        state_matrix = np.array([[1], [0], [0]])
    elif state == 2:
        state_matrix = np.array([[0], [1], [0]])
    else:
        state_matrix = np.array([[0], [0], [1]])

    measurement_state = C @ state_matrix

    #based on the probabilities, randomly pick a measurement
    measurement_state = np.random.choice([1, 2, 3], p=measurement_state.flatten())

    # convert that measurement integer back into a matrix to feed back
    if measurement_state == 1:
        measurement = np.array([[1], [0], [0]])
    elif measurement_state == 2:
        measurement = np.array([[0], [1], [0]])
    else:
        measurement = np.array([[0], [0], [1]])

    ##### END #####
    return measurement

def sim_moodeng(initial_state=1,iteration = 20):
    # Given an initial state of Moo-Deng's whereabout and number of iteration,
    # simulate Moo-Deng's behavior and the designed state estimator
    
    belief = np.array([[1/3], [1/3], [1/3]])  # Initial belief for the Bayesian filter
    
    ##### ADD your code here : #####
    A = np.array([[0.6, 0.2, 0.2],
                  [0.4, 0.4, 0.1],
                  [0.0, 0.4, 0.7]])
    C = np.array([[0.8, 0.2, 0.05], 
                  [0.1, 0.7, 0.1],
                  [0.1, 0.1, 0.85]])
    ##### END #####

    # initalize some variable to use later
    states = []
    measurements = []
    estimated_states = []
    beliefs = []
    state = initial_state

    previous_belief = belief    
    for i in range(iteration):
               
        ##### ADD your code here : #####
        
        #get the simulated true next state
        next_state = moodeng_behavior_update(state, A)
        #based on the next state, get the measurement
        measurement = sensor_measurement(next_state, C)
        
        #calculate the prediction
        prediction_k = A @ previous_belief

        #get the updated belief 
        updated_belief = C.T @ measurement * prediction_k
        nu = 1/np.sum(updated_belief)   #normalizer
        updated_belief = nu * updated_belief

        #pick the estimated state as the state with the highest probability
        estimated_state =np.argmax(updated_belief) +1

        #update for next iteration
        previous_belief = updated_belief
        state = next_state

        ##### END #####
        states.append(next_state)
        measurements.append(measurement)
        beliefs.append(updated_belief)
        estimated_states.append(estimated_state)
        
    return states, measurements, estimated_states, beliefs

    
# Run the simulation

states, measurements, estimated, beliefs = sim_moodeng(initial_state=1, iteration=20)

readings = decode_measurement(measurements)
# Print results
print("True states:          ", [int(x) for x in states])
print("Sensor measurements:  ", [int(x) for x in readings])
print("Estimated states:     ", [int(x) for x in estimated])
print("Belief sequence:      ", np.array(beliefs))