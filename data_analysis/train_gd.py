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

from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

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

    W1 = torch.full((d, d), 1.0, requires_grad=True)  #torch.ones(d, 1, requires_grad=True) #torch.randn(d, 1, requires_grad=True)
    b1 = torch.zeros(d, requires_grad=True)
    W2 = torch.full((d, int(d//2)), 1.0, requires_grad=True)  #torch.ones(d, 1, requires_grad=True) #torch.randn(d, 1, requires_grad=True)
    b2 = torch.zeros(int(d//2), requires_grad=True)

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

    
def train_nonlinear_model(
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
    
    model = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d,int(d//2)), nn.ReLU(), nn.Linear(int(d//2), 1))
    # Optimizer (this replaces manual updates)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)  # (N, 1)

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

    return model