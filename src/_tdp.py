import numpy as np

def accumulate_transitions(predicteds, minimum_separation=0.1):
    transitions = []
    lengths = []
    for predicted in predicteds:
        trans, lens = find_transitions_in_single_trace(predicted, minimum_separation)
        transitions.extend(trans)
        lengths.extend(lens)
    transitions = np.array(transitions)
    return transitions, lengths

def find_transitions_in_single_trace(Z, minimum_separation=0.1):
    transitions = []
    lengths = []
    start = 0
    for i in range(0,len(Z)-1):
        #transitions.append([Z[i],Z[i+1]])
        #if Z[i]!=Z[i+1]:
        if np.abs(Z[i] - Z[i+1])>minimum_separation:
            transitions.append([Z[i],Z[i+1]])
            lengths.append(i-start)
            start = i
    return np.array(transitions), np.array(lengths)
