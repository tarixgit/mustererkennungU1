import numpy as np


def taketrainingdict(trainigfolder):
    #output = {}
    digitarrs = []
    for digit in range(10):
        filename = trainigfolder + 'train.' + str(digit)
        file = open(filename, 'r')

        for line in file:
            arr = list(map(float, line.split(',')))
            digitarrs.append(arr)
        #output[digit] = digitarrs
    return digitarrs


def change_dimension(train_data, n):
    cov = np.cov(train_data, rowvar=False)
    eigenvals, eigenvects = np.linalg.eig(np.mat(cov))
    eigenvals_arr = eigenvals.reshape(len(eigenvals), 1)
    eigenvects_eigenvals = eigenvects_eigenvals.append(eigenvects, eigenvals_arr, axis=1)

    eigenval_indexes = np.argsort(eigenvects_eigenvals, axis=len(eigenvals_arr[0]))
    n_eigenval_indexes = eigenval_indexes[0:n-1:1]
    n_eigenvects = eigenvects[:, n_eigenval_indexes]
    return 0


def pca():
    testfilename = './datasource/test/zip.test'
    trainigfolder = './datasource/training/'
    testfile = open(testfilename, 'r')
    trainingdict = taketrainingdict(trainigfolder)
    trainingarr = np.array(trainingdict)
    change_dimension(trainingarr, 10)
    return 0

pca()