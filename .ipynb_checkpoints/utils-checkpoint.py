import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import random
import pickle
from sklearn.metrics import roc_auc_score

from PIL import Image
from scipy.signal import fftconvolve
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import argparse

from model import load_model
from load_data import load_data_
from train import train_model


def find_bias(p, mlp_head, backbone, test_dataloader, last_layer_dim, device):

    device = 'cpu'
    
    mlp_head.eval()
        
    for batch in test_dataloader:
        X, y = batch
        X.to(device); y.to(device)
        
        with torch.no_grad():
            # Forward pass
            features = backbone(X.float())
            features = features.view(len(X), -1)
            rand_zeros = np. random.randint(0, high=features.shape[1], size=(features.shape[0], int(p*features.shape[1])))
            for i in range(len(X)):
                features[i, rand_zeros[i,:]] = 0
    
        out = mlp_head(features)            
    
        if last_layer_dim == 1:
            valid_loss = F.binary_cross_entropy_with_logits(out.float(), y.float().unsqueeze(1))
            pred = (torch.sigmoid(out) > 0.5).long()
        elif last_layer_dim == 2:
            valid_loss = F.cross_entropy(out.float(), y.float().to(torch.long))    
            pred = out.argmax(1)
            
        final_val_acc = (pred==y).sum()/len(pred)
        final_score = roc_auc_score(y, pred)

    class_stim0 = (pred == 0).sum(); class_stim1 = (pred == 1).sum()

    return final_val_acc, final_score, pred, y, class_stim0, class_stim1, len(X)