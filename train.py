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

def train_model(backbone, mlp_head, train_dataloader, test_dataloader, last_layer_dim, n_epochs, device):

    for param in backbone.parameters():
        param.requires_grad = False

    backbone.eval()

    optimizer = optim.Adam(mlp_head.parameters())

    train_acc = []
    valid_acc = []
    valid_roc_score = []
    valid_loss_arr = []

    for epoch in range(n_epochs):
        # Train
        mlp_head.train()
    
        print("EPOCH", epoch)
    
        for bid, batch in enumerate(train_dataloader):
            
            X, y = batch
            X.to(device); y.to(device)

            with torch.no_grad():
                features = backbone(X.float())
                
            out = mlp_head(features)            
            if last_layer_dim == 1:
                loss = F.binary_cross_entropy_with_logits(out.float(), y.float().unsqueeze(1))
            elif last_layer_dim == 2:
                loss = F.cross_entropy(out.float(), y.float().to(torch.long))
                    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Predict on training data
            with  torch.no_grad():
                if last_layer_dim == 1:
                    pred = (torch.sigmoid(out) > 0.5).long()
                elif last_layer_dim == 2:
                    pred = out.argmax(1)
            
            epoch_train_acc = (pred.squeeze()== y).sum()/len(pred)
            train_acc.append(epoch_train_acc)
    
            print(
                "Epoch {}, Train Batch {}, Loss {:.4f}, Accuracy {:.4f}".format(
                    epoch, bid, loss, epoch_train_acc
                )
            )
          
        # Evaluate
        mlp_head.eval()
    
        for batch in test_dataloader:
            X, y = batch
            X.to(device); y.to(device)
            
            with torch.no_grad():
                # Forward pass
                features = backbone(X.float())
                out = mlp_head(features)            

                if last_layer_dim == 1:
                    valid_loss = F.binary_cross_entropy_with_logits(out.float(), y.float().unsqueeze(1))
                    pred = (torch.sigmoid(out) > 0.5).long()
                elif last_layer_dim == 2:
                    valid_loss = F.cross_entropy(out.float(), y.float().to(torch.long))    
                    pred = out.argmax(1)
                    
                epoch_val_acc = (pred.squeeze()==y).sum()/len(pred)
                score = roc_auc_score(y, pred.squeeze())

                print(
                "VALID Epoch {}, Loss {:.4f}, Accuracy {:.4f}".format(
                    epoch, valid_loss, epoch_val_acc
                )
                )

                valid_acc.append(epoch_val_acc)
                valid_roc_score.append(score)
                valid_loss_arr.append(valid_loss)


        return train_acc, valid_acc, valid_loss_arr, valid_roc_score
