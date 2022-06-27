import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.datasets.samples_generator import make_blobs

def load_data(path):
    data = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(row)
    return data

def process_data(data):
    X = []
    y = []

    for row in data:
        tmp = []
        tmp.append(float(row[0]))
        tmp.append(float(row[1]))
        tmp.append(float(1))
        X.append(tmp)
        y.append([float(row[2])])
    return (np.array(X), np.array(y))

def svm(X, y, w, alpha=0.001, steps=2000, lamb=1):
    loss_array = []
    
    for iter in range(steps):
        loss = np.dot(w.T, w)[0,0]/2
        score = X @ w
        loss +=  np.sum(np.maximum(1-y*score, 0))
        loss_array.append(loss)
        
        grad = -y * (y * score < 1) 
        grad = w + lamb*(X.T @ grad)
        w -= alpha * grad
        if(iter % 1000 == 0):
            continue
        
    return loss_array

def process_for_plot(X, y):
    data = []
    
    counter = 0
    for yi in y:
        data.append((X[counter], yi[0]))
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

if __name__ == "__main__":    
    centers = [(0, -1), (0, 1)]
    cluster_std = [.3, .4]

    X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
    
    X0 = np.ones((100,1))
    X = np.hstack((X,X0))
    y[y == 0] = -1
    y=y.reshape((1,y.size)).T

    w = np.zeros((3,1))
    loss_to_plot = svm(X, y, w)
    data_to_plot = process_for_plot(X, y)
    plot_data(data_to_plot)
    m,b = plot_db(w)
    print("w1: ", w[0][0])
    print("w2: ", w[1][0])
    print("bias: ", w[2][0])
    print("slope: ", m[0])
    print("intercept: ", b[0])

    
    
    
    
    
    
    
    
    
    

        
    
    