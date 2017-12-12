import numpy as np
import matplotlib.pyplot as plt


def taketrainingdict(trainigfolder):
    output = {}
    for digit in range(10):
        filename = trainigfolder + 'train.' + str(digit)
        file = open(filename, 'r')
        digitarrs = []
        for line in file:
            arr = list(map(float, line.split(',')))
            digitarrs.append(arr)
        output[digit] = digitarrs
    return output


def get_eigenvectors(data, n):
    train_data = []
    for i in range(len(data)):
        train_data.extend(data[i])
    cov = np.cov(train_data, rowvar=False)
    eigenvals, eigenvects = np.linalg.eig(np.mat(cov))
    eigenval_indexes = np.argsort(-eigenvals)
    n_eigenval_indexes = eigenval_indexes[0:n]
    n_eigenvects = eigenvects[n_eigenval_indexes]
    return n_eigenvects


def change_dimension(n_eigenvects, train_data):
    new_data = train_data * np.transpose(n_eigenvects)
    return new_data


def visualize(data):
    c_value = ['orange', 'yellow', 'green', 'blue', 'pink', 'black', 'brown', 'purple', 'gray', 'gold']
    for i in range(len(data)):
        a = data[i]
        plt.scatter(a[:, 0], a[:, 1], marker='x', color=c_value[i])
        plt.show()


def pca():
    testfilename = './datasource/test/zip.test'
    trainigfolder = './datasource/training/'
    testfile = open(testfilename, 'r')
    trainingdict = taketrainingdict(trainigfolder)
    new_data = {}
    n_eigenvects = get_eigenvectors(trainingdict, 2)
    for i in range(len(trainingdict)):
        newarr = change_dimension(n_eigenvects, trainingdict[i])
        new_data[i] = newarr
    visualize(new_data)
    return 0

pca()