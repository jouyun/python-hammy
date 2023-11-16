import pandas as pd
import numpy as np
import scipy as sp
import plotly.express as px
from hmmlearn import hmm
import dask

def do_single_fit(values, n_guessed_components, iterations_per_fit = 100):
    guessed_means = (np.arange(0,n_guessed_components)/(n_guessed_components-1)).reshape(-1,1)
    np_values = np.array(values).reshape(-1,1)

    model = hmm.GaussianHMM(n_components = n_guessed_components, covariance_type = "full", n_iter = iterations_per_fit, 
                            init_params="mcs",  means_prior=guessed_means,
                            )
    model.fit(np_values)
    # Predict the hidden states corresponding to observed X.
    Z = model.predict(np_values)
    score = model.score(np.array(values).reshape(-1,1))
    return model, Z, score

def do_aggregate_fit(appended_values, list_of_lengths, n_guessed_components, iterations_per_fit = 100):
    guessed_means = (np.arange(0,n_guessed_components)/(n_guessed_components-1)).reshape(-1,1)
    np_values = np.array(appended_values).reshape(-1,1)

    model = hmm.GaussianHMM(n_components = n_guessed_components, covariance_type = "full", n_iter = iterations_per_fit, 
                            init_params="mcs",  means_prior=guessed_means,
                            )
    model.fit(np_values, list_of_lengths)
    # Predict the hidden states corresponding to observed X.
    Z = model.predict(np_values)
    score = model.score(np.array(appended_values).reshape(-1,1))
    return model, Z, score

def find_best_fit(values, n_guessed_components, num_trials=30):
    best_model = None
    best_score = -np.inf
    best_Z = None

    for i in range(num_trials):
        try:
            model, Z, score = do_single_fit(values, n_guessed_components)
            if score > best_score:
                best_score = score
                best_model = model
                best_Z = Z
            
        except:
            pass
    return best_model, best_model.means_[best_Z].squeeze()

def find_best_fit_combined(list_of_values, n_guessed_components, num_trials=30):
    vals = []
    lens = []
    for values in list_of_values:
        vals.extend(values)
        lens.append(len(values))

    best_model = None
    best_score = -np.inf
    best_Z = None

    for i in range(num_trials):
        try:
            model, Z, score = do_aggregate_fit(vals, lens, n_guessed_components)
            if score > best_score:
                best_score = score
                best_model = model
                best_Z = Z
            
        except:
            pass
    return best_model, best_model.means_[best_Z].squeeze()
                    
    

def dask_wrapper(values, n_guessed_components, num_trials=30):
    model, predicted = find_best_fit(values, n_guessed_components, num_trials)
    return predicted

def fit_all(list_of_values, n_guessed_components, num_trials=30, minimum_gap=0.0001):    
    predicteds = [dask.delayed(dask_wrapper)(values, n_guessed_components, num_trials) for values in list_of_values]    
    predicteds = dask.compute(*predicteds)
    
    returned = []
    for p in predicteds:
        returned.append(cleanup_predictions(p, minimum_gap=minimum_gap))
    return returned


def cleanup_predictions(predictions, minimum_gap=0.1):
    outputs = predictions.copy()
    states, cts = np.unique(predictions, return_counts=True)
    sort_idx = np.argsort(cts)
    
    for cur in np.arange(len(sort_idx)):
        cur_fret = states[sort_idx[cur]]
        for cand in np.arange(cur+1, len(sort_idx)):
            cand_fret = states[sort_idx[cand]]
            if np.abs(cur_fret-cand_fret) < minimum_gap:
                outputs[outputs==cur_fret] = cand_fret
    return outputs
        
    