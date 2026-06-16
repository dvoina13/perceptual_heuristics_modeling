import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import random
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from sklearn import svm
from scipy.linalg import solve

import torch
import torch.nn as nn
import torch.optim as optim

def compute_cp(student_w, student_b, features, y, tunings, crit, n_neurons, choose_y = "model", animal_choice = None, min_per_choice=6, min_wrong=3, force_ge_half=False, eps=1e-8, selected_trials = None):

    # features: (T, N) or (1,T,N) etc -> make (T, N)
    X = features.squeeze()
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    y = np.asarray(y.squeeze())
    n_stim = 2

    # logits + choice
    out = (student_w @ X.T + student_b)
    
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    out = np.asarray(out).ravel()

    threshold = 0.0   #if crit == "MSE" else 0.0
    choice = (out >= threshold).astype(int)

    if crit != "BCE":
        choice = 2*choice - 1

    if choose_y != "model":
        choice = animal_choice

    print("choice", choice)
        
    neurons_tuned_stim1, neurons_tuned_stim0, neurons_untuned = tunings
    neurons_tuned_list = list(neurons_tuned_stim1) + list(neurons_tuned_stim0)

    # sample up to 1000 neurons
    #rng = np.random.default_rng(0)
    #if len(neurons_tuned_list) > n_neurons:
    #    neurons_tuned_list = list(rng.choice(neurons_tuned_list, size=n_neurons, replace=False))

    # allocate by LOOP INDEX (safe)
    CP_arr = np.full((len(neurons_tuned_list), n_stim), np.nan)
    new_tunings = []

    list_of_stim = [0, 1] if crit == "BCE" else [-1, 1]
    
    for i, n in enumerate(neurons_tuned_list):

        pref = (0 if crit == "BCE" else -1) if n in neurons_tuned_stim0 else (1)            
        new_tunings.append(pref)

        for s in list_of_stim:
            
            print("n, s", n, s)
            idx = np.where(y == s)[0]
            print("idx", idx)
            
            if idx.size == 0:
                continue

            r_raw = X_np[idx, n]                  # raw firing rates this condition
 
            # FIX 4: z-score within condition to remove mean stimulus drive
            mu, sd = r_raw.mean(), r_raw.std()
            r = (r_raw - mu) / (sd + eps) if sd > eps else r_raw - mu
            #r = r_raw
            
            o = (choice[idx] == pref).astype(int)  # labels: 1 = preferred choice
                
            n1 = int(o.sum())
            n0 = int((1 - o).sum())
            print("choice is preferred, or unpreferred", n0, n1)
            wrong = np.sum(choice[idx] != y[idx])
            #wrong = min(n0, n1)  # proxy if you don't have correctness labels

            if n1 < min_per_choice or n0 < min_per_choice or wrong < min_wrong:
                print("OH nooo, the classifier i just too good and doesn't make mistakes")
                continue

            if selected_trials!=None:
                ind_selected_trials = list(set(idx) & set(selected_trials))
            
            print("o", "r", o, r)
            auc = roc_auc_score(o, r)  # <-- correct order
            print("auc", auc, 1-auc)
            if force_ge_half:
                auc = max(auc, 1 - auc)

            s_idx = 0 if s in (-1, 0) else 1
            CP_arr[i, s_idx] = auc

    cp = np.nanmean(CP_arr, axis=1)  # mean across stim, per sampled neuron
    return cp, CP_arr, neurons_tuned_list, new_tunings

def compute_noise_corr(features, y, neurons, zclip=3.0, eps=1e-8):
    X = np.asarray(features.squeeze())[:, neurons]
    stim = np.asarray(y).ravel()

    Z = np.empty_like(X, dtype=float)

    for s in np.unique(stim):
        idx = np.where(stim == s)[0]
        mu = X[idx].mean(axis=0)
        sd = X[idx].std(axis=0) + eps
        Z[idx] = (X[idx] - mu) / sd

    keep = np.all(np.abs(Z) <= zclip, axis=1)
    Z = Z[keep]

    # Drop neurons with zero variance BEFORE corrcoef
    neuron_std = Z.std(axis=0)
    valid = neuron_std > eps
    Z = Z[:, valid]
     
    if Z.shape[1] < 2:
        # fallback: return identity
        return np.eye(Z.shape[1]), Z, valid

    C = np.corrcoef(Z, rowvar=False)
    return C, Z, valid  # <-- also return valid mask

        
def compute_readout_weights(student_w, student_b, features, y, tunings, crit, n_neurons, choose_y = "model", animal_choice = None, eps=1e-7, normalize = False, selected_trials = None):

    cp, CP_arr, neurons_tuned_list, new_tunings = compute_cp(student_w, student_b, features, y, tunings, crit, n_neurons, choose_y, animal_choice, selected_trials = selected_trials)
    C, Z, valid = compute_noise_corr(features, y, neurons_tuned_list)    
    
    C = np.asarray(C, dtype=float)
    cp = np.asarray(cp, dtype=float).ravel()[valid]
    
    # NaN/inf safety net before solve
    assert np.all(np.isfinite(C)), "C still has NaN/inf"
    if not np.all(np.isfinite(cp)):
        return "NaN"
    
    diag = np.clip(np.diag(C), eps, None)
    rhs = (np.pi * np.sqrt(diag) * (cp - 0.5)) / np.sqrt(2)
    
    # Solve C beta = v  (preferred over inv(C) @ v)
    try:
        beta = solve(C, rhs, assume_a="sym")
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(C, rhs, rcond=None)
    
    if normalize:
        denom = np.sqrt(beta.T @ C @ beta) + eps
        beta = beta / denom

    return beta



def find_bias_simple(p, student_w, student_b, features, y, crit="BCE", comparison="heuristic"):
        print("p", p)
        mean = features.clone().mean(0).numpy()
        features_sparse = features.squeeze().numpy().copy() #- mean
        m, n = features_sparse.shape
        
        for i in range(m):
            rand_zeros = np.random.choice(n, size=int(p*n), replace=False)
            features_sparse[i,rand_zeros] = 0
            #print(list(set(rand_zeros) & set(neurons_circ_new)), student_w[list(set(rand_zeros) & set(neurons_circ_new))], features[i,list(set(rand_zeros) & set(neurons_circ_new))], student_w[list(set(rand_zeros) & set(neurons_circ_new))].dot(features[i,list(set(rand_zeros) & set(neurons_circ_new))]))

        #features_sparse = features_sparse - mean #features_sparse.mean(0)
    
        logits = torch.matmul(torch.from_numpy(features_sparse).float(), student_w.float()) + student_b
        logits_nonsparse = torch.matmul(features.float(), student_w.float()) + student_b
    
        y = y.squeeze(); logits = logits.squeeze(); logits_nonsparse = logits_nonsparse.squeeze();
        
        if comparison == "heuristic":
            ind0 = np.where(y == 0)[0] 
            ind1 = np.where(y == 1)[0]
        else:
            print('hello')
            ind0 = np.where(y == -1)[0] 
            ind1 = np.where(y == 1)[0] 

        if crit == "BCE":
            ind0 = np.where(y == 0)[0] 
            ind1 = np.where(y == 1)[0] 

        y_res = logits.detach().numpy() - student_b.item()

        print("criteria is: ", crit)
        if crit == "BCE":
            threshold = 0.0
            acc1 = ( (y[logits < threshold] == 0).sum() + (y[logits > threshold] == 1).sum() )/len(y)
            acc2 = ( (y[logits > threshold] == 0).sum() + (y[logits < threshold] == 1).sum() )/len(y)

            acc1_nonsparse = ( (y[logits_nonsparse < threshold] == 0).sum() + (y[logits_nonsparse > threshold] == 1).sum() )/len(y)
            acc2_nonsparse = ( (y[logits_nonsparse > threshold] == 0).sum() + (y[logits_nonsparse < threshold] == 1).sum() )/len(y)

        elif crit == "MSE":
            threshold = 0.0
            acc1 = ( (y[logits < threshold] == -1).sum() + (y[logits > threshold] == 1).sum() )/len(y)
            acc2 = ( (y[logits > threshold] == -1).sum() + (y[logits < threshold] == 1).sum() )/len(y)

            acc1_nonsparse = ( (y[logits_nonsparse < threshold] == -1).sum() + (y[logits_nonsparse > threshold] == 1).sum() )/len(y)
            acc2_nonsparse = ( (y[logits_nonsparse > threshold] == -1).sum() + (y[logits_nonsparse < threshold] == 1).sum() )/len(y)

        print("acc1, acc2", acc1, acc2)
        if acc1 == acc2:
            print("given p=", str(p) + ", then accuracy is: ", acc2)
            return acc2, (logits>threshold).sum()
            
        if acc1_nonsparse < acc2_nonsparse: #student_w.mean()< 0: #
            print("given p=", str(p) + ", then accuracy is: ", acc2, "class 1 labels (under-represented)", (logits>=threshold).sum(), "class 2 labels (over-represented)", (logits<threshold).sum())
            print("means of classes: ", logits[ind0].mean(), logits[ind1].mean())
            print("std of classes: ", logits[ind0].std(), logits[ind1].std())
            return acc2, (logits>=threshold).sum()
        else:
            print("given p=", str(p) + ", then accuracy is: ", acc1, "class 1 labels (under-represented)", (logits<=threshold).sum(), "class 2 labels (over-represented)", (logits>threshold).sum())
            print("means of classes: ", logits[ind0].mean(), logits[ind1].mean())
            print("std of classes: ", logits[ind0].std(), logits[ind1].std())
            return acc1, (logits<=threshold).sum()


def compute_other_cp_inactivation(writer, X_train, y_train, tunings, model):

    neurons_circ_new, neurons_rad_new, neurons_untuned_new = tunings
    
    if model == "SUM":
        student_w = torch.ones(X_train.shape[1])
        student_b = -X_train.mean(0).sum()

        out = (student_w @ X_train.T + student_b)
    else:

        n_neurons = X_train.shape[1]
        student_w = np.zeros(n_neurons)
        
        student_w[neurons_circ_new] = 0.1
        student_w[neurons_rad_new] = -0.1

        out = student_w @ X_train.T
        student_b = - out.mean()
        out += student_b
        
    ind0 = np.where(y_train == 0)[0]
    ind1 = np.where(y_train == 1)[0]

    #plot_results(writer, out, ind0, ind1)

    print("acc: ", float( (out[ind0] <= 0).sum() + (out[ind1] >= 0).sum() ) / len(out))

    class_imbalance = []
    p_inactivation = [0.0001, 0.3, 0.5, 0.75, 0.9]
    for p in p_inactivation:
        acc, under_rep_imbalance = find_bias_simple(p, torch.tensor(student_w).T, torch.tensor(student_b), torch.tensor(X_train).float(), y_train)
        class_imbalance.append(under_rep_imbalance)

    #plot_choice_imbalance(writer, p_inactivation, class_imbalance)

    crit = "BCE"
    cp, CP_arr, neurons_tuned_list, new_tunings = compute_cp(student_w, student_b, X_train, y_train, tunings, crit, len(student_w))
    readout_weights = compute_readout_weights(student_w, student_b, X_train, y_train, tunings, crit, len(student_w))

    #plot_cp_rw(writer, readout_weights, tunings)

    return out, ind0, ind1, p_inactivation, class_imbalance, readout_weights

def save_all(file, session, eigenvector, student_w, tunings, accuracy_train_theuristic, accuracy_heuristic, cp, cp_real, readout_weights, readout_weights_real, class_imbalance, cp_train, cp_real_train, readout_weights_train, readout_weights_real_train, class_imbalance_train, readout_weights_real_total, cp_real_total, readout_weights_sum, class_imbalance_sum, readout_weights_diff, class_imbalance_diff):

    np.save("results/" + file + "session_" + str(session) + "_eigenvector_vh.npy", eigenvector)
    np.save("results/" + file + "session_" + str(session) + "_weight.npy", eigenvector)
    np.save("results/" + file + "session_" + str(session) + "_acc.npy", accuracy_heuristic)
    np.save("results/" + file + "session_" + str(session) + "_acc_training.npy", accuracy_train_theuristic)
    
    np.save("results/" + file  + "session_" + str(session) + "_cp.npy", cp)
    np.save("results/" + file  + "session_" + str(session)+ "_cp_real.npy", cp_real)

    np.save("results/" + file  + "session_" + str(session) + "_readout_weights.npy", readout_weights)
    np.save("results/" + file  + "session_" + str(session) + "_readout_weights_real.npy", readout_weights_real)
    np.save("results/" + file  + "session_" + str(session) + "_readout_weights_sum.npy", readout_weights_sum)
    np.save("results/" + file  + "session_" + str(session) + "_readout_weights_diff.npy", readout_weights_diff)

    np.save("results/" + file  + "session_" + str(session) + "_class_imbalance.npy", class_imbalance)
    np.save("results/" + file  + "session_" + str(session) + "_class_imbalance_sum.npy", class_imbalance_sum)
    np.save("results/" + file  + "session_" + str(session) + "_class_imbalance_diff.npy", class_imbalance_diff)

    np.save("results/" + file  + "session_" + str(session) + "_readout_weights_train.npy", readout_weights_train)
    np.save("results/" + file  + "session_" + str(session) + "_readout_weights_real_train.npy", readout_weights_real_train)
    np.save("results/" + file  + "session_" + str(session) + "_cp_train.npy", cp_train)
    np.save("results/" + file  + "session_" + str(session) + "_cp_real_train.npy", cp_real_train)
    np.save("results/" + file  + "session_" + str(session) + "_class_imbalance_train.npy", class_imbalance_train)

    np.save("results/" + file  + "session_" + str(session) + "_readout_weights_real_total.npy", readout_weights_real_total)
    np.save("results/" + file  + "session_" + str(session) + "_cp_real_total.npy", cp_real_total)
    
    np.save("results/" + file  + "session_" + str(session) + "_neurons_tuned_circ.npy", np.array(tunings[0]))
    np.save("results/" + file  + "session_" + str(session) + "_neurons_tuned_rad.npy", np.array(tunings[1]))
    np.save("results/" + file  + "session_" + str(session) + "_neurons_untuned.npy", np.array(tunings[2]))