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

def transform_data(data):
    train_data = []
    for i in range(len(data)):
        train_data.extend(data[i])
    return train_data

def get_eigenvectors(data, n):
    cov = np.cov(data, rowvar=False)
    eigenvals, eigenvects = np.linalg.eig(np.mat(cov))
    eigenval_indexes = np.argsort(-eigenvals)
    n_eigenval_indexes = eigenval_indexes[0:n]
    n_eigenvects = eigenvects[n_eigenval_indexes]
    return n_eigenvects


def change_dimension(n_eigenvects, train_data):
    new_data = train_data * np.transpose(n_eigenvects)
    return new_data


def visualize(data):
    fig = plt.figure(figsize=(40, 20))
    ax = [fig.add_subplot(3, 4, 1), fig.add_subplot(3, 4, 2), fig.add_subplot(3, 4, 3), fig.add_subplot(3, 4, 4),
          fig.add_subplot(3, 4, 5),
          fig.add_subplot(3, 4, 6), fig.add_subplot(3, 4, 7), fig.add_subplot(3, 4, 8), fig.add_subplot(3, 4, 9),
          fig.add_subplot(3, 4, 10)]
    c_value = ['orange', 'yellow', 'green', 'blue', 'pink', 'black', 'brown', 'purple', 'gray', 'gold']
    for i in range(len(data)):
        a = data[i]
        ax[i].scatter(a[:, 0], a[:, 1], color=c_value[i], label=i, s=3)
    plt.show()


def pca():
    testfilename = './datasource/test/zip.test'
    trainigfolder = './datasource/training/'
    testfile = open(testfilename, 'r')
    trainingdict = taketrainingdict(trainigfolder)
    rawtrainingdict = transform_data(trainingdict)
    n_eigenvects = get_eigenvectors(rawtrainingdict, 2)

    new_data = {}
    for i in range(len(trainingdict)):
        newarr = change_dimension(n_eigenvects, trainingdict[i])
        new_data[i] = newarr
    visualize(new_data)
    return 0

pca()