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
from utils import find_bias
from save_data import save_data_

parser = argparse.ArgumentParser(description="args for training the neural network")
parser.add_argument("--seed", type=int, default = 1, help="seed")
parser.add_argument("-batchsize", type=int, default = 50, help="batch size during training")
parser.add_argument("-layer", type=int, default = 7, help="resnet layer to use for classification")
parser.add_argument("-epochs", type=int, default = 1, help="number of epochs during training")
parser.add_argument("-last_layer_dim", type=int, default = 1, help="output layer dim")
parser.add_argument("-lr", type=float, default = 0.01, help="learning rate")
parser.add_argument("-lr_bool", type=bool, default = False, help="learning rate, to be or not to be")
parser.add_argument("-optimization", type=str, default = "Adam", help="Adam, SGD, or others")
parser.add_argument("--p", type=float, default = 0.5, help="How much to lesion the last year")

args = parser.parse_args()

seed = args.seed
batchsize = args.batchsize
layer_to_use = args.layer
n_epochs = args.epochs
last_layer_dim = args.last_layer_dim
p = args.p

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = 'cpu'


train_dataloader, test_dataloader = load_data_(seed, batchsize)
backbone, mlp_head = load_model(layer_to_use, last_layer_dim)
train_acc, valid_acc, valid_loss_arr, valid_roc_score = train_model(backbone, mlp_head, train_dataloader, test_dataloader, last_layer_dim, n_epochs, device)

final_val_acc, final_score, pred, y, class_stim0, class_stim1, len_test = find_bias(p, mlp_head, backbone, test_dataloader, last_layer_dim, device)

save_data_(seed, p, mlp_head, valid_acc, valid_loss_arr, valid_roc_score, final_val_acc, final_score, pred, y, batchsize, layer_to_use, last_layer_dim)

print(f"Final valid acc is {final_val_acc}, with stim labeled class 0: {class_stim0} / {len_test}, with stim labeled class 1:  with stim labeled class 1: {class_stim1} / {len_test}")