import numpy as np


#load the data by one
def load_data(filename):
    file = open(filename, 'r')
    digitarrs = []
    for line in file:
        arr = list(map(float, line.split(',')))
        digitarrs.append(arr)
    file.close()
    return digitarrs


# separate training data and rest data (80/20%)
def split_arr(arr):
    train_data = arr[:int((len(arr) + 1) * .80)]  # Remaining 80% to training set
    test_data = arr[int(len(arr) * .80 + 1):]  # Splits 20% data to test set
    return train_data, test_data


def normalization(arr):
    nor_arr = []
    lenth = len(arr)
    label_arr = arr[lenth:]
    for i in range(lenth - 1):
        mean = np.mean(arr[:i])
        var = np.var(arr[:i])
        nor_arr.append((arr[:i] - mean)/var)
    nor_arr = np.transpose(nor_arr)
    for i in range(lenth):
        nor_arr[i:].append(label_arr[i])
    return nor_arr


def sigmoid(beta, arr_x):
    lenth = len(arr_x)
    p = (1/1+np.exp(np.dot(beta, arr_x[0:lenth - 1]) * (-arr_x[lenth])))
    return p


def logistic_regression():
    filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    arr = load_data(filename)
