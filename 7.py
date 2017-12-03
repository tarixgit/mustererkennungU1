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
    m, n = np.shape(arr)
    x0 = np.ones(m)
    nor_arr = [x0]
    label_arr = arr[:, n - 1]
    for i in range(n - 1):
        xi = arr[:, i] #-1?
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
    return 1.0/(1 + np.exp(np.dot(beta, x_vector) * (-y)))

def find_beta(train_data, beta0, gamma, iteration_number):
    #
    beta_current = beta0
    train_data_len = len(train_data)
    vector_len = len(train_data[0])
    for i in range(iteration_number):
        #get dldb
        dldb = np.zeros(vector_len - 1)
        for j in range(train_data_len):
            xi = train_data[j][:vector_len - 1]
            y = train_data[j][vector_len - 1]
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
    data_length, vector_len = np.shape(test_data)
    test_data_full = np.ones((data_length, vector_len + 1))
    test_data_full[:, 1:] = test_data

    #data_length = len(test_data)
    #vector_len = len(test_data[0])
    for i in range(data_length):
        y = np.dot(beta, test_data_full[i][:vector_len])
        if test_data_full[i][vector_len] == 1:
            all1 += 1
            if y < 0:
                error1 += 1
        else:
            all2 += 1
            if y >= 0:
                error2 += 1
    print("p: " + str(all1) + " error 1: " + str(error1) + ", eee: " + str(error1/all1) + ", n: " + str(all2) + " error2: " + str(error2)+ ", eee: " + str(error2/all2))


def logistic_regression():
    filename = './datasource/spambase.data'
    arr = load_data(filename)
    arr = np.array(arr)
    #data_arr = normalization(arr)
    train_data, test_data = split_arr(arr)
    train_data_norm = normalization(train_data)
    beta0 = intiliaze_beta(train_data_norm)
    gamma = 10 ** (-5)
    iteration_number = 300
    beta = find_beta(train_data_norm, beta0, gamma, iteration_number)
    classificator(beta, test_data)



logistic_regression()