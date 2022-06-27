import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

def generate_data(centers, stds, n=100, number_features=2):
    centers = centers
    cluster_std = stds

    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=number_features, random_state=1)
    
    X0 = np.ones((100,1))
    X = np.hstack((X,X0))
    y[y == 0] = -1
    y=y.reshape((1,y.size)).T
    
    return X, y

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

def log_reg(X, y, w, alpha=0.001, steps=2000):
    loss_array = []
    for iter in range(steps):
        tmp = np.matmul(X, w)
        tmp = -y * tmp
        tmp = np.exp(tmp)

        loss = np.log(1+tmp)
        loss = np.mean(loss)

        loss_array.append(loss)

        g = -y * tmp
        g = g / (1 + tmp)
        g = np.matmul(X.T, g)
        
        w = w - alpha*g
    return w, loss_array


if __name__ == "__main__":
    X, y = generate_data([(0, -.25), (0, .75)], [.3, .4])
    w = np.zeros((3,1))
    w, loss_to_plot = log_reg(X, y, w)
    data_to_plot = process_for_plot(X, y)
    plot_data(data_to_plot)
    m,b = plot_db(w)
    print("w1: ", w[0][0])
    print("w2: ", w[1][0])
    print("bias: ", w[2][0])
    print("slope: ", m[0])
    print("intercept: ", b[0])
    
    