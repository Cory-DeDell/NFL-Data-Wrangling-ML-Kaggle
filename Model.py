# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:38:12 2022

@author: Cory
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
import seaborn as sns
from Utils import *

def main():
    # Load the training and testing data
    X_train = pd.read_csv('X_train.csv', low_memory=False).to_numpy()
    X_val = torch.tensor(pd.read_csv('X_val.csv', low_memory=False).to_numpy(), dtype=torch.float32)
    y_train = pd.read_csv('y_train.csv', low_memory=False).to_numpy()
    y_val = torch.tensor(pd.read_csv('y_val.csv', low_memory=False).to_numpy(), dtype=torch.float32)
    
    
    kf = KFold(n_splits = 5)
    train_fold_losses = []
    test_fold_losses = []
    models = []
    
    for train_idx, test_idx in kf.split(X_train, y_train):
        x_tr, y_tr = X_train[train_idx], y_train[train_idx]
        x_tst, y_tst = X_train[test_idx], y_train[test_idx]
        
        # batch data
        batch_size = 32
        train_batches = batch_data(x_tr, y_tr, batch_size)
        test_batches = batch_data(x_tst, y_tst, batch_size)
    
        #################################
        # Model specification
        model = nn.Sequential(
                  nn.Dropout(p=0.5),
                  nn.Linear(495, 1024),
                  nn.BatchNorm1d(1024),
                  nn.ReLU(),
                  nn.Dropout(p=0.5),
                  nn.Linear(1024, 512),
                  nn.BatchNorm1d(512),
                  nn.ReLU(),
                  nn.Dropout(p=0.5),
                  nn.Linear(512, 256),
                  nn.BatchNorm1d(256),
                  nn.ReLU(),
                  nn.Dropout(p=0.5),
                  nn.Linear(256, 199))
        lr = 0.001
        momentum = 0

        ##################################
    
        train_loss, test_loss = train_model(train_batches, test_batches, model, lr=lr, momentum=momentum)
        models.append(model)
        train_fold_losses.append(train_loss)
        test_fold_losses.append(test_loss)
        
    avg_train_loss = np.mean(train_fold_losses)
    avg_test_loss = np.mean(test_fold_losses)
    
    # evauluate models on validation set
    for i,model in enumerate(models):
        model.eval()
        y_pred = model(X_val)[1][-1].detach().numpy()

if __name__ == '__main__':
    np.random.seed(12345)
    torch.manual_seed(12345)
    main()
