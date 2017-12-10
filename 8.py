import numpy as np


def take_training_dict(trainig_folder):
    output = {}
    for digit in range(10):
        filename = trainig_folder + 'train.' + str(digit)
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
    new_data = train_data * np.transpose(n_eigenvects)
    return new_data


def pca():
    # test_filename = './datasource/test/zip.test'
    trainig_folder = './datasource/training/'
    # test_file = open(test_filename, 'r')
    training_dict = take_training_dict(trainig_folder)
    vectors = []
    vector = []
    for i in range(len(training_dict)):
        n_eigenvects = change_dimension(training_dict[i])
        for j in range(len(n_eigenvects)):
            for k in range(len(n_eigenvects[0])):
                vector.append(n_eigenvects[j][k])
            vectors.append(vector)
    return 0