import numpy as np


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


def change_dimension(train_data, n):
    cov = np.cov(train_data, rowvar=0)
    eigenvals, eigenvects = np.linalg.eig(np.mat(cov))
    eigenval_indexes = np.argsort(-eigenvals)
    n_eigenval_indexes = eigenval_indexes[0:n-1:1]
    n_eigenvects = eigenvects[:, n_eigenval_indexes]
    return 0


def pca():
    testfilename = './datasource/test/zip.test'
    trainigfolder = './datasource/training/'
    testfile = open(testfilename, 'r')
    trainingdict = taketrainingdict(trainigfolder)
    return 0