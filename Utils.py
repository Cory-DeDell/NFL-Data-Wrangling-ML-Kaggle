"""Training utilities."""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def batch_data(X, y, batch_size):
    """Takes a set of data points and labels and groups them into random batches."""

    batches = []
    permutation = torch.randperm(X.shape[0])

    for i in range(0, X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        batches.append({
            'x': torch.tensor(X[indices], dtype=torch.float32),
            'y': torch.tensor(y[indices], dtype=torch.long).flatten()+99})
        
    return batches

def train_model(train_data, test_data, model, lr=0.01, momentum=0.9, n_epochs=30):
    """Train a model for n_epochs given data and params."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    training_losses = []
    test_losses = []
    epochs = np.arange(1,n_epochs+1)
    
    for epoch in range(n_epochs):
        print("-------------\nEpoch {}:\n".format(epoch+1))

        # Run **training***
        train_loss = run_epoch(train_data, model.train(), optimizer)
        print('Train loss: {:.6f}'.format(train_loss))

        # Run **test**
        test_loss = run_epoch(test_data, model.eval(), optimizer)
        print('Test loss:   {:.6f}'.format(test_loss))
        
        training_losses.append(train_loss)
        test_losses.append(test_loss)
        
    sns.lineplot(x=epochs,y=training_losses)
    sns.lineplot(x=epochs,y=test_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['Training', 'Test'])
    plt.title("Traing and Test loss vs epoch")
    plt.show()
    
    return train_loss, test_loss

def run_epoch(data, model, optimizer):
    """Train model for one pass of train dat and return loss"""
    # Gather losses
    losses = []
    
    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x and y
        x, y = batch['x'], batch['y']

        # Get output predictions
        out = model(x)

        # Compute loss
        loss = F.cross_entropy(out,y)
        # print(loss)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate mean epoch loss
    avg_loss = np.mean(losses)

    return avg_loss
