import numpy as np
import matplotlib.pyplot as plt
import csv

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

def log_reg(X, y, w, alpha=0.00001, steps=1000):
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
    data = load_data('data.csv')
    X, y = process_data(data)
    w = np.array([[0,0,0]]).T
    optimal_w, loss_array = log_reg(X, y, w)
    print(optimal_w)
    plt.plot(loss_array)
    plt.show()
    