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
    # ax = [fig.add_subplot(3, 4, 1), fig.add_subplot(3, 4, 2), fig.add_subplot(3, 4, 3), fig.add_subplot(3, 4, 4),
    #       fig.add_subplot(3, 4, 5),
    #       fig.add_subplot(3, 4, 6), fig.add_subplot(3, 4, 7), fig.add_subplot(3, 4, 8), fig.add_subplot(3, 4, 9),
    #       fig.add_subplot(3, 4, 10)]
    ax = []
    c_value = ['orange', 'yellow', 'green', 'blue', 'pink', 'black', 'brown', 'purple', 'gray', 'gold']
    count=1
    for i in range(len(data)):
        a = data[i]
        for j in range(i+1, len(data)):
            b = data[j]
            x = fig.add_subplot(5, 9, count)
            x.plot(a[:, 0], a[:, 1], 'r+', color=c_value[i], label=i)
            x.plot(b[:, 0], b[:, 1], 'r+', color=c_value[j], label=j)
            x.set_title(str(i) + ' and ' + str(j))
            ax.append(x)
            count += 1
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