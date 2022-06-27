# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
def generate_data(centers, stds, n=100, number_features=2):
    centers = centers
    cluster_std = stds
    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=number_features, random_state=1)
    y[y == 0] = -1    
    return X, y

def process_for_plot(X, y):
    data = []
    
    counter = 0
    for yi in y:
        data.append((X[counter], yi))
        counter += 1
    return data

def plot_data(data):
    for data_to_plot_i in data:
        x1 = data_to_plot_i[0][0]
        x2 = data_to_plot_i[0][1]
        yi = data_to_plot_i[1]
        if(yi == 1):
            plt.scatter(x1, x2, color='b')
        elif(yi==-1):
            plt.scatter(x1, x2, color='r')
    plt.grid(True)   
    
def plot_db(w):
    m=-1*(w[0]/w[1])
    b=-1*(w[2]/w[1])
    x = np.linspace(-4,4,100)
    y = m*x+b
    plt.plot(x, y, '-r', label='Decision boundary')
    return m, b

def plot_data_with_db(model, X_nmp, y_nmp):
    w = model.fc1.weight.data
    b = model.fc1.bias.item()
    w=[w[0][0],w[0][1],b]

    data_to_plot = process_for_plot(X_nmp, y_nmp)
    plot_data(data_to_plot)
    m,b = plot_db(w)
    
def generate_data_tensors(centers, stds):
    X_nmp, y_nmp = generate_data(centers, stds)
    X = X_nmp.T
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y_nmp).float()
    return X, y, X_nmp, y_nmp

class SVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_array = []
        self.fc1 = nn.Linear(2,1,bias=True)
        
    def forward(self, x):
        return self.fc1(x)

    def svm_trainer(self, X, y, C, learning_rate = 0.1, steps = 1000):
        optimizer = optim.SGD(self.parameters(), lr = learning_rate)
        
        for iter in range(steps):
            w = self.fc1.weight
            output = model(X.t())
            optimizer.zero_grad()
            loss = (C/2) * (torch.matmul(w, w.t())) + torch.sum(torch.clamp(1 - y.view(100,1)*output, min=0))
            self.loss_array.append(loss.data.item())
            loss.backward()
            optimizer.step()

LEARNING_RATE = .01
STEPS = 1000
C = .0001

X, y, X_nmp, y_nmp = generate_data_tensors([(0, -.9), (0, 1.2)], [.3, .4])

model = SVM()
model.svm_trainer(X, y, C, LEARNING_RATE, STEPS)
plot_data_with_db(model, X_nmp, y_nmp)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    