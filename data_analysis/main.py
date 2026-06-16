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

from create_data import run
from train_Oja import compute_svd, train_oja_unsupervised
from utils import compute_cp, compute_readout_weights, find_bias_simple, compute_other_cp_inactivation, save_all
from plot import plot_weights, plot_cp_rw, plot_choice_imbalance, plot_results

from torch.utils.tensorboard import SummaryWriter

import parameters
from parameters import *

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cpu'
n_sessions, data_dictionary, df_all_sessions, tunings, tunings_session, X_train, y_train, animal_choice_train, X_test, y_test, animal_choice_test, X_total, y_total, animal_choice_total, selected_trials =run(directory, snr_choice = snr_choice, choice=choice, session = session, get_data="generate") #get_data="saved"
selected_trials = None

Vh = compute_svd(X_train)

if choice == "only_session":
    tunings = tunings_session

student_w, student_b, eta = train_oja_unsupervised(torch.tensor(X_train).float(), add_bias=centering, centering=centering)
eigenvector = np.sign(Vh[0,:].mean())*Vh[0,:]
#student_w = np.sign(student_w.mean())*torch.tensor(eigenvector)
#student_b = -student_b if student_w.mean()<0 else student_b
#student_w = -student_w if student_w.mean()<0 else student_w

writer = SummaryWriter("runs/my_experiment_" + directory + "_seed_" + str(seed) + "_snr_choice_" + str(snr_choice) + "_choice_" + choice + ["_centering" if centering else ""][0])
plot_weights(writer, student_w, student_b, Vh, tunings)

ind0 = np.where(y_train == 0)[0]  
ind1 = np.where(y_train == 1)[0]

y_exp = torch.matmul(torch.tensor(X_train).float(), torch.tensor(student_w).squeeze().float()) + student_b
y_exp = y_exp.detach().numpy()
accuracy_train_heuristic1 =  float( (y_exp[ind0] <= 0).sum() + (y_exp[ind1] >= 0).sum() ) / len(y_exp)

y_exp = torch.matmul(torch.tensor(X_train).float(), torch.tensor(-student_w).squeeze().float()) - student_b
y_exp = y_exp.detach().numpy()
accuracy_train_heuristic2 =  float( (y_exp[ind0] <= 0).sum() + (y_exp[ind1] >= 0).sum() ) / len(y_exp)

if accuracy_train_heuristic1 < accuracy_train_heuristic2:
    student_w = -student_w
    student_b = -student_b
    accuracy_train_heuristic = accuracy_train_heuristic2
else:
    accuracy_train_heuristic = accuracy_train_heuristic1

print("accuracy is: ", accuracy_train_heuristic)

### compute RW/CP on testing data
features = X_test #X_train #- X_train.mean(0) 
y = y_test
a_choice = animal_choice_test

y_exp = torch.matmul(torch.tensor(features).float(), torch.tensor(student_w).squeeze().float()) + student_b
y_exp = y_exp.detach().numpy()
ind0 = np.where(y == 0)[0]  
ind1 = np.where(y == 1)[0]

accuracy_heuristic =  float( (y_exp[ind0] <= 0).sum() + (y_exp[ind1] >= 0).sum() ) / len(y_exp)
print("(test) accuracy is: ", accuracy_heuristic)

crit = "BCE"
n_neurons = X_train.shape[1]
cp, CP_arr, neurons_tuned_list, new_tunings = compute_cp(torch.tensor(student_w).float(), student_b, torch.tensor(features).float(), y, tunings, crit, n_neurons, selected_trials = selected_trials)
readout_weights = compute_readout_weights(torch.tensor(student_w).float(), student_b, torch.tensor(features).float(), y, tunings, crit, n_neurons, selected_trials = selected_trials)

if type(readout_weights) != str:
    print("readout weights: ", [readout_weights[:len(tunings[0])].mean(), readout_weights[len(tunings[0]):].mean()])
plot_cp_rw(writer, readout_weights, tunings)

cp_real, CP_arr_real, neurons_tuned_list, new_tunings = compute_cp(torch.tensor(student_w).float(), torch.tensor(student_b), torch.tensor(features).float(), y, tunings, crit, n_neurons, choose_y = "real", animal_choice=a_choice)
readout_weights_real = compute_readout_weights(torch.tensor(student_w).float(), torch.tensor(student_b), torch.tensor(features).float(), y, tunings, crit, n_neurons, choose_y = "real", animal_choice=a_choice)

if type(readout_weights_real) != str:
    print("readout weights for actual data: ", [readout_weights_real[:len(tunings[0])].mean(), readout_weights_real[len(tunings[0]):].mean()])

plot_cp_rw(writer, readout_weights, tunings)

class_imbalance = []
p_inactivation = [0.0001, 0.3, 0.5, 0.75, 0.9]
for p in p_inactivation:
    acc, under_rep_imbalance = find_bias_simple(p, torch.tensor(student_w).T.float(), student_b, torch.tensor(features).float(), y)
    class_imbalance.append(under_rep_imbalance)

print("class imbalance: ", class_imbalance)

plot_choice_imbalance(writer, p_inactivation, class_imbalance)

print("OTHER weights... (for sum and diff models)")

### compute RW/CP on training data
features = X_train #- X_train.mean(0) 
y = y_train
a_choice = animal_choice_train

crit = "BCE"
n_neurons = X_train.shape[1]
cp_train, CP_arr, neurons_tuned_list, new_tunings = compute_cp(torch.tensor(student_w).float(), student_b, torch.tensor(features).float(), y, tunings, crit, n_neurons, selected_trials = selected_trials)
readout_weights_train = compute_readout_weights(torch.tensor(student_w).float(), student_b, torch.tensor(features).float(), y, tunings, crit, n_neurons, selected_trials = selected_trials)

cp_real_train, CP_arr_real, neurons_tuned_list, new_tunings = compute_cp(torch.tensor(student_w).float(), torch.tensor(student_b), torch.tensor(features).float(), y, tunings, crit, n_neurons, choose_y = "real", animal_choice=a_choice)
readout_weights_real_train = compute_readout_weights(torch.tensor(student_w).float(), torch.tensor(student_b), torch.tensor(features).float(), y, tunings, crit, n_neurons, choose_y = "real", animal_choice=a_choice)

class_imbalance_train = []
p_inactivation = [0.0001, 0.3, 0.5, 0.75, 0.9]
for p in p_inactivation:
    acc, under_rep_imbalance = find_bias_simple(p, torch.tensor(student_w).T.float(), student_b, torch.tensor(features).float(), y)
    class_imbalance_train.append(under_rep_imbalance)

cp_real_total, CP_arr_real, neurons_tuned_list, new_tunings = compute_cp(torch.tensor(student_w).float(), torch.tensor(student_b), torch.tensor(X_total).float(), y_total, tunings, crit, n_neurons, choose_y = "real", animal_choice=animal_choice_total)
readout_weights_real_total = compute_readout_weights(torch.tensor(student_w).float(), torch.tensor(student_b), torch.tensor(X_total).float(), y_total, tunings, crit, n_neurons, choose_y = "real", animal_choice=animal_choice_total)

### compute RW/CP for sum and diff models
out_sum, ind0_sum, ind1_sum, p_inactivation, class_imbalance_sum, readout_weights_sum = compute_other_cp_inactivation(writer, X_train, y_train, tunings, model="SUM")
plot_results(writer, out_sum, ind0_sum, ind1_sum)
plot_choice_imbalance(writer, p_inactivation, class_imbalance_sum)
plot_cp_rw(writer, readout_weights_sum, tunings)

out_diff, ind0_diff, ind1_diff, p_inactivation, class_imbalance_diff, readout_weights_diff = compute_other_cp_inactivation(writer, X_train, y_train, tunings, model="DIFF")
plot_results(writer, out_diff, ind0_diff, ind1_diff)
plot_choice_imbalance(writer, p_inactivation, class_imbalance_diff)
plot_cp_rw(writer, readout_weights_diff, tunings)

#save stuff
file = directory[:-1] + "_seed_" + str(seed) + "_snr_choice_" + str(snr_choice) + "_choice_" + choice + ["_centering" if centering else ""][0]
save_all(file, session, Vh[0,:], student_w, tunings, accuracy_train_heuristic, accuracy_heuristic, cp, cp_real, readout_weights, readout_weights_real, class_imbalance, cp_train, cp_real_train, readout_weights_train, readout_weights_real_train, class_imbalance_train,  readout_weights_real_total, cp_real_total, readout_weights_sum, class_imbalance_sum, readout_weights_diff, class_imbalance_diff)
