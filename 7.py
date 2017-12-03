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
    train_data = np.arr[:int((len(arr) + 1) * .80)]  # Remaining 80% to training set
    test_data = np.arr[int(len(arr) * .80 + 1):]  # Splits 20% data to test set
    return train_data, test_data

def intiliaze_beta(train_data):
    random.seed()
    x = random.randint(0,  len(train_data) - 1)
    beta = train_data[x]
    return beta[:len(beta) - 1]

def do_beta_step(beta_before, gamma, dldb):
    beta_after = beta_before + gamma * dldb
    return beta_after

def get_posibillity():
    return


def normalization(arr):
    mean = np.mean(arr[])


def logistic_regression:
    filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    arr = load_data(filename)
    train_data, test_data = split_arr(arr)
    beta0 = intiliaze_beta(train_data)


logistic_regression()