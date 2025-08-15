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

from load_data import generate_all_imgs

def load_model(layer_to_use, last_layer_dim):
    # Load the pre-trained ResNet50 model
    resnet50 = models.resnet50(pretrained=True)
    
    # Set the model to evaluation mode (important for inference)
    resnet50.eval()
    
    # If you want to use a GPU, move the model to the GPU
    if torch.cuda.is_available():
        resnet50 = resnet50.cuda()
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean= [0.485, 0.456, 0.406], #[0.493, 0.493, 0.493],
                             std=[0.229, 0.224, 0.225]) #[0.1075, 0.1075, 0.1075])#)
        ])
    
    
    all_imgs = generate_all_imgs()
    x = all_imgs[0]
    x = transform(Image.fromarray(x))
    sizes = []
    
    for i, child in enumerate(list(resnet50.children())):
        if i == 0:
            x = child(torch.tensor(x).unsqueeze(0).float())
        else:
            try:
                x = child(x)
            except:
                x = child(x.squeeze())
        sizes.append(x.shape)
    
        
    backbone = nn.Sequential(*list(resnet50.children())[:layer_to_use])
    
    backbone_output_size = np.prod(sizes[layer_to_use - 1])
    mlp_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(backbone_output_size, last_layer_dim),
    )

    return backbone, mlp_head
    
    
    
