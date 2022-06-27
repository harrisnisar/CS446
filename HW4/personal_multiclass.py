# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
def generate_data(centers, stds, n=100, number_features=2):
    centers = centers
    cluster_std = stds

    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=number_features, random_state=1)
    
#    X0 = np.ones((100,1))
#    X = np.hstack((X,X0))

    y=y.reshape((1,y.size)).T
    
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

        if(yi[0] == 1.):
            plt.scatter(x1, x2, color='b')
        elif(yi[1] == 1.):
            plt.scatter(x1, x2, color='r')
        elif(yi[2] == 1.):
            plt.scatter(x1, x2, color='g')
        elif(yi[3] == 1.):
            plt.scatter(x1, x2, color='y')
    plt.grid(True)   
    
def plot_db(w, color):
    m=-1*(w[0]/w[1])
    b=-1*(w[2]/w[1])
    x = np.linspace(-4,4,100)
    y = m*x+b
    plt.plot(x, y, color, label='Decision boundary')
    return m, b

def one_hot_encode(y):
    eye_mat = np.eye(4)
    y_to_return=[]
    for yi in y:
        y_to_return.append(eye_mat[yi][0])
    return np.array(y_to_return)

def generate_data_tensors(centers, stds):
    X_nmp, y_nmp = generate_data(centers, stds)
    y_nmp = one_hot_encode(y_nmp)
    X = X_nmp
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y_nmp).float()
    return X, y, X_nmp, y_nmp



class multiclass_logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_array = []
        self.fc1 = nn.Linear(2, 4, bias=True)
        
    def forward(self, x):
        return self.fc1(x)

if __name__ == "__main__":
    X, y, X_nmp, y_nmp = generate_data_tensors([(2, -2), (-2, -2), (0, 2)], [.13, .2,.09])
#    print(y_nmp)
    data_to_plot = process_for_plot(X_nmp, y_nmp)
#    plot_data(data_to_plot)

   
    model = multiclass_logistic()
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    for iter in range(1000):
        optimizer.zero_grad()
        loss = model.forward(X)
#        prediction = F.softmax(prediction)
        loss = F.log_softmax(loss)
        loss = y * loss
        loss = torch.sum(loss, dim=1)
        loss = -torch.sum(loss)
        model.loss_array.append(loss.data.item())
        loss.backward()
        optimizer.step()
#        print(loss.data.item())
#    plt.plot(model.loss_array)
#    print(model.fc1.bias.data)
    color = ['-b','-r','-g']
    plot_data(data_to_plot) 
    for i in range(3):
        w = model.fc1.weight[i].data
        w_array = [w[0].item(),w[1].item(), model.fc1.bias[i].data.item()]
        plot_db(w_array, color[i])
    plt.axis([-5, 5, -5, 5])
    print(w_array)
#    plot_db(model.fc1)
        

    


    
    
