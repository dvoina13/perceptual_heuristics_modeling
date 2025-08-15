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

def generate_all_imgs():
    all_imgs = []

    n = 0
    for i in range(12):
        for j in range(12):

            img = cv2.imread("/home/dvoina/projects/rrg-shahabkb/dvoina/postdoc_umontreal/project1_heuristics/Pooya_data/Image_set_Lavern/" + str(n + 1) + ".png")
            all_imgs.append(img)
            n += 1

    return all_imgs

def gaussian_3d_kernel(size_t, size_x, size_y, sigma_t, sigma_x, sigma_y):
    
    t = np.arange(-size_t//2 + 1, size_t//2 + 1)
    x = np.arange(-size_x//2 + 1, size_x//2 + 1)
    y = np.arange(-size_y//2 + 1, size_y//2 + 1)
    
    T, X, Y = np.meshgrid(t, x, y, indexing='ij')
    
    kernel = np.exp(-(T**2 / (2*sigma_t**2) +
                      X**2 / (2*sigma_x**2) +
                      Y**2 / (2*sigma_y**2)))
    return kernel / np.sum(kernel)


def gaussian_2d_kernel(size_x, size_y, sigma_x, sigma_y):
    
    x = np.arange(-size_x//2 + 1, size_x//2 + 1)
    y = np.arange(-size_y//2 + 1, size_y//2 + 1)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    kernel = np.exp(-(X**2 / (2*sigma_x**2) +
                      Y**2 / (2*sigma_y**2)))
    return kernel / np.sum(kernel)



class Stimuli():
    
    def __init__(self, img, frames, type_of_image=None):
        self.image = []
        self.type_of_image = type_of_image
        
        for f in range(frames):
            self.image.append(torch.tensor(img))
        self.image = np.stack(self.image)
            
    def add_noise(self, seed):
        torch.manual_seed(seed)
        self.noisy_image = []
        
        white_noise = torch.randn_like(torch.tensor(self.image), dtype=torch.float32)
        for c in range(3):
            kernel_3d = gaussian_3d_kernel(self.image.shape[0], self.image.shape[1], self.image.shape[2], 0.1, 0.1, 0.1)
            correlated_noise = fftconvolve(white_noise[:,:,:,c], kernel_3d, mode='same')
            correlated_noise -= correlated_noise.mean()
            correlated_noise /= correlated_noise.std()

            self.noisy_image.append(self.image[:,:,:,c]+ 1*correlated_noise.astype(int))
            print('correlated_noise', correlated_noise.astype(int))
            self.correlated_noise = correlated_noise
            
        self.noisy_image = np.stack(self.noisy_image)
        self.noisy_image = np.transpose(self.noisy_image, (1,2,3,0))



def load_data_(seed, batchsize):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    all_imgs = generate_all_imgs()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean= [0.485, 0.456, 0.406], #[0.493, 0.493, 0.493],
                             std=[0.229, 0.224, 0.225]) #[0.1075, 0.1075, 0.1075])#)
        ])
    
    
    indices = list(range(48,56)) + list(range(136,144))
    images_of_interest = []
    for i in indices:
        images_of_interest.append(all_imgs[i])
    
    
    n_train = 1000
    n_test = 100
    
    train_set_x = []
    test_set_x = []
    train_set_y = []
    test_set_y = []
    
    for n in range(n_train):
        choice = random.choice(list(range(len(indices))))
        img = images_of_interest[choice]
    
        kernel_2d = gaussian_2d_kernel(img.shape[0], img.shape[1], 1, 1)
        for c in range(3):
            white_noise = torch.randn_like(torch.tensor(img[:,:,c]), dtype=torch.float32)
    
            correlated_noise = fftconvolve(white_noise, kernel_2d, mode='same')
            correlated_noise -= correlated_noise.mean()
            correlated_noise /= correlated_noise.std()
    
            img[:,:,c] = img[:,:,c] + correlated_noise
    
        img = transform(Image.fromarray(img))
        #img = np.resize(img, (224,224,3));
        
        train_set_x.append(img)
        if choice <=7:
            train_set_y.append(0)
        else:
            train_set_y.append(1)
    
    for n in range(n_test):
        choice = random.choice(list(range(len(indices))))
        img = images_of_interest[choice]
    
        kernel_2d = gaussian_2d_kernel(img.shape[0], img.shape[1], 1, 1)
        for c in range(3):
            white_noise = torch.randn_like(torch.tensor(img[:,:,c]), dtype=torch.float32)
    
            correlated_noise = fftconvolve(white_noise, kernel_2d, mode='same')
            correlated_noise -= correlated_noise.mean()
            correlated_noise /= correlated_noise.std()
    
            img[:,:,c] = img[:,:,c] + correlated_noise
    
        img = transform(Image.fromarray(img))
        #img = np.resize(img, (224,224,3));
        
        test_set_x.append(img)
    
        if choice <=7:
            test_set_y.append(0)
        else:
            test_set_y.append(1)
    
    
    training_data_x = torch.tensor(np.stack(train_set_x))
    testing_data_x = torch.tensor(np.stack(test_set_x))
    training_data_y = torch.tensor(train_set_y)
    testing_data_y = torch.tensor(test_set_y)
    
    training_data = TensorDataset(training_data_x, training_data_y)
    testing_data = TensorDataset(testing_data_x, testing_data_y)
    
    
    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=len(test_set_y))
    
    return train_dataloader, test_dataloader