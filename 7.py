import numpy as np
import random as random

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
    #TODO: check if this work for list
    np.random.shuffle(arr)
    train_data = arr[:int((len(arr) + 1) * .80)]  # Remaining 80% to training set
    test_data = arr[int(len(arr) * .80 + 1):]  # Splits 20% data to test set
    return train_data, test_data

def intiliaze_beta(train_data):
    random.seed()
    x = random.randint(0,  len(train_data) - 1)
    beta = train_data[x]
    return beta[:len(beta) - 1]

def do_beta_step(beta_before, gamma, dldb):
    beta_after = beta_before + gamma * dldb
    return beta_after

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


def sigmoid_p(beta, x_vector, y): #possibility function
    return 1.0/1 + np.exp(np.dot(beta, x_vector) * (-y))

def find_beta(beta0, gamma, train_data_arr):
    #
    beta_current = beta0
    length = len(train_data_arr)
    for i in range(length - 1):
        if sigmoid(beta_current, vector, y) = 0:
            dldb = 0.....
    beta_after = do_beta_step(beta_current, gamma, dldb)
    return True;

def logistic_regression():
    filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    arr = load_data(filename)
    train_data, test_data = split_arr(arr)
    train_data_arr = np.array(train_data)
    beta0 = intiliaze_beta(train_data)
    train_data_norm = normalization(train_data_arr)

logistic_regression()