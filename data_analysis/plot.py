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

from utils import compute_cp, compute_readout_weights, find_bias_simple
from torch.utils.tensorboard import SummaryWriter

def plot_results(writer, out, ind0, ind1):
    fig = plt.figure()
    plt.plot(out[ind0])
    plt.plot(out[ind1])
    plt.xlabel("trials")
    plt.ylabel("output")
    plt.title("Assess class separabilty and success of learning rule")
    writer.add_figure("plots/out_class0_vs_class1", fig, global_step=0)

def plot_weights(writer, weights, b, Vh, tunings):

    neurons_circ_new, neurons_rad_new, neurons_untuned_new = tunings
    
    weights = weights.squeeze()

    fig = plt.figure()
    plt.plot(weights.detach().numpy())
    plt.plot(-Vh[0,:])
    plt.title("w versus PC1 for various tuned neurons")
    plt.xlabel("weight index")
    plt.ylabel("weight")
    plt.legend(["w", "PC1"])
    writer.add_figure("plots/w_vs_v", fig, global_step=0)

    fig = plt.figure()
    if list(neurons_circ_new) != []:
        plt.plot(weights.detach().numpy()[np.array(neurons_circ_new)], "*-")
    if list(neurons_rad_new) != []:
        plt.plot(weights.detach().numpy()[np.array(neurons_rad_new)], "*-")
    if list(neurons_untuned_new) != []:
        plt.plot(weights.detach().numpy()[np.array(neurons_untuned_new)], "*-")
    plt.legend(["circular neurons", "radial neurons", "untuned neurons"])
    plt.title("w for various tuned neurons")
    plt.xlabel("weight index")
    plt.ylabel("weight")
    writer.add_figure("plots/weights_circ_rad_untuned", fig, global_step=0)
    
    if list(neurons_circ_new) != []:
        fig = plt.figure()
        plt.plot(weights.detach().numpy()[np.array(neurons_circ_new)], "*-")
        plt.plot(-Vh[0,np.array(neurons_circ_new)], "*-")
        plt.title("w versus PC1 for circular tuned neurons")
        plt.legend(["w", "PC1"])
        plt.xlabel("weight index")
        plt.ylabel("weight")
        writer.add_figure("plots/weights_vs_v_circ", fig, global_step=0)
        
    if list(neurons_rad_new) != []:
        fig = plt.figure()
        plt.plot(weights.detach().numpy()[np.array(neurons_rad_new)], "*-")
        plt.plot(-Vh[0,np.array(neurons_rad_new)], "*-")
        plt.title("w versus PC1 for radial tuned neurons")
        plt.legend(["w", "PC1"])
        plt.xlabel("weight index")
        plt.ylabel("weight")
        writer.add_figure("plots/weights_vs_v_rad", fig, global_step=0)

    writer.close()

    print("mean weights: ", weights[np.array(neurons_circ_new)].mean(), weights[np.array(neurons_rad_new)].mean(), weights[np.array(neurons_untuned_new)].mean())
    print("std weights: ", weights[np.array(neurons_circ_new)].std(), weights[np.array(neurons_rad_new)].std(), weights[np.array(neurons_untuned_new)].std())
    print("total mean: ", weights.mean())                                                                                                                 
def plot_cp_rw(writer, readout_weights, tunings):

    if type(readout_weights) == str:
        fig = plt.figure()
        plt.title("readout weights are NaN (non-computable)")
        writer.add_figure("plots/bars_readout_weights", fig, global_step=0)

        writer.close()
        return 
        
    fig = plt.figure()
    plt.bar(["stim 1 (over-represented)", "stim 2 (under-represented)"], [readout_weights[:len(tunings[0])].mean(), readout_weights[len(tunings[0]):].mean()])
    plt.title("Readout weights")
    writer.add_figure("plots/bars_readout_weights", fig, global_step=0)

    writer.close()
    
def plot_choice_imbalance(writer, p_inactivation, class_imbalance):
    
    fig = plt.figure()
    plt.plot(p_inactivation, class_imbalance, "*-")
    plt.xlabel("probability", fontsize=16)
    plt.ylabel("class imbalance", fontsize=16)
    plt.title("Bias upon inactivation")
    writer.add_figure("plots/bars_readout_weights", fig, global_step=0)

    writer.close()
    