import numpy as np
import matplotlib.pyplot as plt


class ReductionDimension:
    def taketrainingdict(self, trainigfolder):
        # output = {}
        digitarrs = []
        for digit in range(10):
            filename = trainigfolder + 'train.' + str(digit)
            file = open(filename, 'r')

            for line in file:
                arr = list(map(float, line.split(',')))
                digitarrs.append(arr)
                # output[digit] = digitarrs
        self.trainingarr = np.array(digitarrs)

    def change_dimension(self, train_data, n):
        cov = np.cov(train_data, rowvar=False)
        eigenvals, eigenvects = np.linalg.eig(np.mat(cov))
        eigenval_indexes = np.argsort(-eigenvals)
        n_eigenval_indexes = eigenval_indexes[0:n]
        n_eigenvects = eigenvects[n_eigenval_indexes]
        new_data = train_data * np.transpose(n_eigenvects)
        self.new_data = new_data
        return new_data

def visualize(data):
    plt.figure(figsize=(12, 8))
    plt.plot(data[:,0], data[:,1], 'r+',)
    plt.show()

def pca():
    testfilename = './datasource/test/zip.test'
    trainigfolder = './datasource/training/'
    testfile = open(testfilename, 'r')
    dim = ReductionDimension()
    dim.taketrainingdict(trainigfolder)
    newarr = dim.change_dimension(dim.trainingarr, 10)
    visualize(newarr)
    return 0

pca()