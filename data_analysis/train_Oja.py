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

def compute_svd(X_train):

    X = X_train  #- X_train.mean(0)
    cov = X.T @ X
    U, s, Vh = np.linalg.svd(X)

    return Vh

def train_oja_unsupervised(features, student_w_init = None, add_bias=False, centering=False):

    n_samples = features.shape[0]
    n_total = features.shape[1]
    rand_ind = np.random.permutation(np.arange(n_samples))
    features = features[rand_ind, :];
    if centering:
        features = features - features.mean(0)
    
    if student_w_init is not None:
        student_w = student_w_init
    else:
        student_w = torch.ones(1, n_total) * 0.01 #torch.randn(1, n_total) * 0.01
        
    print("student_w", student_w)
    student_b = torch.ones(1)
    eta = 0.0001
    print("eta", eta, student_w.mean())
    
    acc_arr = []; 
    print("Student learning via Oja's Rule...")
    with torch.no_grad():
        for j in range(1000):
            #print("j", j)
            y_arr = []
            for i in range(features.shape[0]):
                x_i = features[i].squeeze().unsqueeze(0)

                if add_bias:
                    y_student = torch.matmul(student_w, x_i.T)[0].item() + student_b
                else:
                     y_student = torch.matmul(student_w, x_i.T)[0].item()
                dw = eta * (y_student * x_i - (y_student**2) * student_w)
                student_w += dw

                
                #learn bias
                y_new = torch.matmul(student_w, x_i.T)[0].item() + student_b
                ybar = y_new #(1 - alpha) * ybar + alpha * y_new
                student_b += eta*(-ybar) #y_target=0


                if np.isnan(student_w.sum()):
                    print("hello")
                    return

                #if i%100:
                #    print("student_b", student_b)
    #y_new = torch.matmul(student_w, features.T)
    #student_b = y_new.mean()
    return student_w, student_b, eta

def train_oja_unsupervised_gd(features, y_true, student_w_init = None, add_bias=False):

    features = torch.as_tensor(features, dtype=torch.float32)
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    
    n_samples = features.shape[0]
    n_total = features.shape[1]
    rand_ind = np.random.permutation(np.arange(n_samples))
    features = features[rand_ind, :];
    y_true = y_true[rand_ind]
    features = features - features.mean(0)
    
    if student_w_init is not None:
        student_w = student_w_init
    else:
        student_w = torch.randn(1, n_total) * 0.01 #torch.ones(1, n_total) * 0.01
        
    student_b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    eta = 0.1
    total_loss = 0
    
    optimizer = optim.Adam([student_b], lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    
    acc_arr = []; 
    print("Student learning via Oja's Rule...")
    #with torch.no_grad():
    for j in range(1000):
            print("j", j)
            y_arr = []

            for i in range(features.shape[0]):
                x_i = features[i].squeeze().unsqueeze(0)

                with torch.no_grad():
                    y_oja = torch.matmul(student_w, x_i.T).squeeze().item()

                    if add_bias:
                        y_student = y_oja + student_b
                    else:
                        y_student = y_oja
                        
                    dw = eta * (y_oja * x_i - (y_oja**2) * student_w)
                    student_w += dw
                    
                optimizer.zero_grad()

                logit = torch.matmul(student_w, x_i.T).squeeze() + student_b
                target = y_true[i].squeeze()

                # target must be 0/1 float for BCE
                loss = bce(logit.view(1), target.view(1))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if np.isnan(student_w.detach().numpy().sum()):
                    print("hello")
                    return

                #if i%100:
                #    print("student_b", student_b)
    #y_new = torch.matmul(student_w, features.T)
    #student_b = y_new.mean()
    return student_w, student_b, eta

def train_gd(features, y_true, student_w_init = None, add_bias=False):

    features = torch.as_tensor(features, dtype=torch.float32)
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    
    n_samples = features.shape[0]
    n_total = features.shape[1]
    rand_ind = np.random.permutation(np.arange(n_samples))
    features = features[rand_ind, :];
    y_true = y_true[rand_ind]
    features = features - features.mean()
    
    if student_w_init is not None:
        student_w = nn.Parameter(student_w_init, requires_grad=True)
    else:
        student_w = nn.Parameter(torch.randn(1, n_total) * 0.01)
        
    student_b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    eta = 0.0001
    total_loss = 0
    
    optimizer = optim.Adam([student_w, student_b], lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    
    acc_arr = []; 
    print("Student learning via Oja's Rule...")
    #with torch.no_grad():
    for j in range(1000):
            print("j", j)
            y_arr = []

            for i in range(features.shape[0]):
                x_i = features[i].squeeze().unsqueeze(0)

                y_pred = torch.matmul(student_w, x_i.T).squeeze().item() + student_b
                target = y_true[i].squeeze()

                optimizer.zero_grad()

                # target must be 0/1 float for BCE
                loss = bce(y_pred.view(1), target.view(1))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if np.isnan(student_w.detach().numpy().sum()):
                    print("hello")
                    return

                #if i%100:
                #    print("student_b", student_b)
    #y_new = torch.matmul(student_w, features.T)
    #student_b = y_new.mean()
    return student_w, student_b

    
