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
n_sessions, data_dictionary, df_all_sessions, tunings, tunings_session, X_train, y_train, animal_choice_train, X_test, y_test, animal_choice_test =run(directory, snr_choice = snr_choice, choice=choice, get_data="generate")

Vh = compute_svd(X_train)

y_train = 2*y_train - 1
y_test = 2*y_test - 1

def train_linear_model(
    X,
    y,
    lr=1e-2,
    n_epochs=50,
    loss_type="mse",  # "mse" or "bce"
    verbose=True,
):
    """
    Train a linear model y = XW + b using gradient descent (optimizer-based).
    """

    N, d = X.shape

    # Ensure y has shape (N, 1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    loss_bce = nn.BCELoss()
    
    # Initialize parameters
    W = torch.full((d, 1), 1.0, requires_grad=True)  #torch.ones(d, 1, requires_grad=True) #torch.randn(d, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # Optimizer (this replaces manual updates)
    optimizer = torch.optim.SGD([W, b], lr=lr)

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = X @ W + b  # (N, 1)

        if loss_type == "mse":
            loss = torch.mean((y_pred - y) ** 2)

        elif loss_type == "bce":
            
            y_pred_sigmoid = torch.sigmoid(y_pred)
            loss = loss_bce(y_pred_sigmoid, y)
                    #torch.mean(-y * torch.log(y_pred_sigmoid + 1e-8) - (1 - y) * torch.log(1 - y_pred_sigmoid + 1e-8))

        else:
            raise ValueError("loss_type must be 'mse' or 'bce'")

        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    return W.detach(), b.detach()


#X_train = X_train.detach().numpy()
#y_train = y_train.detach().numpy()

w_gd, b_gd = train_linear_model(torch.tensor(X_train).float(), torch.tensor(y_train), lr=1e-4, n_epochs=10000, loss_type="mse",  # "mse" or "bce"
    verbose=True)

w_gd = torch.tensor(w_gd); b_gd = torch.tensor(b_gd); 
y = torch.tensor(X_train).float() @ w_gd + b_gd
y = 2*(y.squeeze()>0.0).numpy().astype(float) - 1 #(y.squeeze()>0.0).numpy().astype(float)

ind0 = np.where(y_train == -1)[0] #np.where(y_train == 0)[0]
ind1 = np.where(y_train == 1)[0]

print("loss: ", np.abs(torch.tensor(y).squeeze()-y_train.squeeze()).sum()/len(y_train))
print("acc: ", float( (y[ind0] <= 0).sum() + (y[ind1] >= 0).sum() ) / len(y))

y = torch.tensor(X_test).float() @ w_gd + b_gd
y = 2*(y.squeeze()>0.0).numpy().astype(float) - 1 #(y.squeeze()>0.0).numpy().astype(float)

ind0 = np.where(y_test == -1)[0] #np.where(y_train == 0)[0]
ind1 = np.where(y_test == 1)[0]

print("loss: ", np.abs(torch.tensor(y).squeeze()-y_test.squeeze()).sum()/len(y_test))
print("acc: ", float( (y[ind0] <= 0).sum() + (y[ind1] >= 0).sum() ) / len(y))

neurons_circ_new, neurons_rad_new, neurons_untuned_new = tunings
neurons_circ_new = list(neurons_circ_new); neurons_rad_new = list(neurons_rad_new); neurons_untuned_new = list(neurons_untuned_new);

fig = plt.figure()
if neurons_circ_new != []:
    plt.plot(w_gd.squeeze()[np.array(neurons_circ_new)], "*-")
if neurons_rad_new != []:
    plt.plot(w_gd.squeeze()[np.array(neurons_rad_new)], "*-")
plt.ylabel("Weights", fontsize=16)
plt.xlabel("weight index", fontsize=16)

print(w_gd.squeeze()[np.array(neurons_circ_new)].mean().item(), w_gd.squeeze()[np.array(neurons_rad_new)].mean().item())

writer = SummaryWriter("runs/my_GD_experiment_" + directory + "_seed_" + str(seed) + "_snr_choice_" + str(snr_choice) + "_choice_" + choice + ["_centering" if centering else ""][0])

if neurons_circ_new != []:
    fig = plt.figure()
    plt.plot(w_gd.squeeze()[np.array(neurons_circ_new)])
    #plt.plot(student_w.squeeze()[np.array(neurons_circ_new)], "*-")
    plt.plot(-Vh[0,np.array(neurons_circ_new)], "*-")
    plt.legend(["w gd (circular)", "pc1 (circular)"])
    writer.add_figure("plots/w_gd_circ", fig, global_step=0)

if neurons_rad_new != []:
    fig = plt.figure()
    plt.plot(w_gd.squeeze()[np.array(neurons_rad_new)])
    #plt.plot(student_w[np.array(neurons_rad_new)], "*-")
    plt.plot(-Vh[0,np.array(neurons_rad_new)], "*-")
    plt.legend(["w gd (circular)", "pc1 (circular)"])
    writer.add_figure("plots/w_gd_rad", fig, global_step=0)

crit = "MSE"
n_neurons = X_train.shape[1]
#tunings = [np.array(neuron_id_tuned_circular)-1, np.array(neuron_id_tuned_radial)-1, np.array(neuron_id_tuned_untuned)-1]
tunings = [np.array(neurons_circ_new), np.array(neurons_rad_new), np.array(neurons_untuned_new)]

cp, CP_arr, neurons_tuned_list, new_tunings = compute_cp(w_gd.T, b_gd, X_test, y_test, tunings, crit, n_neurons)
readout_weights = compute_readout_weights(w_gd.T, b_gd, X_test, y_test, tunings, crit, n_neurons)

fig = plt.figure()
plt.bar(["stim 1 (over-represented)", "stim 2 (under-represented)"], [np.nanmean(readout_weights[:len(tunings[0])]), np.nanmean(readout_weights[len(tunings[0]):])])
writer.add_figure("plots/readout_weights_GD", fig, global_step=0)

class_imbalance = []
p_inactivation = [.00001, 0.1, 0.3, 0.5, 0.75, 0.9]
crit = "MSE"
for p in p_inactivation:
    acc, ci = find_bias_simple(p, torch.tensor(w_gd).float(), torch.tensor(b_gd).float(), torch.tensor(X_test).float(), torch.tensor(y_test).float(), crit, comparison="gd")
    class_imbalance.append(ci)

fig = plt.figure()
plt.plot(p_inactivation, class_imbalance, "*-")
plt.xlabel("probability of inactivation")
plt.ylabel("class imbalance (under-represented)")
writer.add_figure("plots/class_imbalance_GD", fig, global_step=0)

file = directory[:-1] + "_GD_seed_" + str(seed) + "_snr_choice_" + str(snr_choice) + "_choice_" + choice + ["_centering" if centering else ""][0]

np.save('results/' + file + "_cp.npy", np.array(cp))
np.save('results/' + file + "_readout_weights.npy", np.array(readout_weights))
np.save('results/' + file + "_class_imbalance.npy", np.array(class_imbalance))

np.save('results/' + file + 'w_gd.npy', np.array(w_gd))
np.save('results/' + file + 'b_gd.npy', np.array(b_gd))

