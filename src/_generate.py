# This file is for generating test trajectories and saving them to files for testing

import numpy as np
import pandas as pd

def generate_transition_matrix(n_states):
    transition_matrix = np.random.random(size=(n_states,n_states)) * 0.05
    for i in range(n_states):
        transition_matrix[i,i] = 1.0 - np.sum(transition_matrix[i,:]) + np.sum(transition_matrix[i,i])
    return transition_matrix

def get_emission_value(state, state_means, state_stds):
    return np.random.normal(loc=state_means[state], scale=state_stds[state])

def generate_trace(n_steps, state_means, state_stds, transition_matrix):
    states = [0]
    values = [get_emission_value(states[-1], state_means, state_stds)]
    shift = np.random.random() * .05 - .025
    for i in range(0,n_steps-1):
        current = states[-1]
        new_state = np.random.choice(np.arange(0,len(state_means)), p=transition_matrix[current])
        states.append(new_state)
        values.append(shift + get_emission_value(new_state, state_means, state_stds))
    return states, values

def generate_data(state_means = [0, 0.5, 1], stdev = 0.1, n_trajectories = 100, trajectory_length = 500, directory = 'data'):
    n_states = len(state_means)
    transition_matrix = generate_transition_matrix(n_states)
    state_stds = np.array([stdev] * len(state_means))

    states = []
    values = []
    for i in range(0,n_trajectories):
        states_sub, values_sub = generate_trace(trajectory_length, state_means, state_stds, transition_matrix)
        pd.DataFrame({'time':np.arange(len(states_sub)),'value': values_sub}).to_csv(directory + '/Trace' + str(i).zfill(3) + '.csv')
        pd.DataFrame({'time':np.arange(len(states_sub)),'value': values_sub}).to_csv(directory + '/Truth' + str(i).zfill(3) + '.csv')
