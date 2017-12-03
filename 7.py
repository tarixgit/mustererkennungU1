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
    m, n = np.shape(arr)
    label_arr = arr[:, n - 1]
    for i in range(n - 1):
        xi = arr[:, i-1] #-1?
        mean = np.mean(xi)
        var = np.var(xi)
        nor_arr.append((xi - mean)/var)
    for i in range(m - 1):
        if label_arr[i] == 0:
            label_arr[i] = -1
    nor_arr.append(label_arr)
    nor_arr = np.transpose(nor_arr)
    return nor_arr


def sigmoid_p(beta, x_vector, y): #possibility function
    return 1.0/1 + np.exp(np.dot(beta, x_vector) * (-y))

def find_beta(beta0, gamma, train_data, iteration_number):
    #
    beta_current = beta0
    train_data_len = len(train_data)
    vector_len = len(train_data[0])
    for i in range(iteration_number):
        #get dldb
        dldb = np.zeros(vector_len)
        for j in range(train_data_len):
            xi = train_data[j][:vector_len - 1]
            y = train_data[j][vector_len]
            correct_vec = y * xi * (1 - sigmoid_p(beta_current, xi, y))
            dldb = dldb + correct_vec
        dldb_step = dldb * gamma
        beta_current = do_beta_step(beta_current, gamma, dldb_step)
        #if beta_after - beta-current < threshold:
        #    break
    return beta_current

def classificator(beta, test_data):
    all1 = 0.0
    error1 = 0.0
    all2 = 0.0
    error2 = 0.0
    length = len(test_data)
    for i in range(length):
        y = np.dot(beta, test_data[:length- 1])
        if test_data[length- 1]==1:
            all1 += 1
            if y<0:
                error1 += 1
        else:
            all2 += 1
            if y>=0:
                error2 += 1
    print("p, all vectors: " + str(all1) + " error 1: " + str(error1) + ", n, all vectors: " + str(all2) + " error2: " + str(error2))


def logistic_regression():
    filename = './datasource/spambase.data'
    arr = load_data(filename)
    arr = np.array(arr)
    data_arr = normalization(arr)
    train_data, test_data = split_arr(data_arr)



logistic_regression()