import numpy as np
import random as random
from sklearn.metrics import confusion_matrix


# load the data by one
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
        xi = arr[:, i]
        mean = np.mean(xi)
        var = np.var(xi)
        nor_arr.append((xi - mean)/var)
    for i in range(m - 1):
        if label_arr[i] == 0:
            label_arr[i] = -1
    nor_arr.append(label_arr)
    nor_arr = np.transpose(nor_arr)
    return nor_arr


# possibility function
def sigmoid_p(beta, x_vector, y):
    return 1.0/(1 + np.exp(np.dot(beta, x_vector) * (-y)))


def find_beta(train_data, beta0, gamma, iteration_number):
    #
    beta_current = beta0
    train_data_len = len(train_data)
    vector_len = len(train_data[0])
    for i in range(iteration_number):
        # get dldb
        dldb = np.zeros(vector_len - 1)
        for j in range(train_data_len):
            xi = train_data[j][:vector_len - 1]
            y = train_data[j][vector_len - 1]
            correct_vec = y * xi * (1 - sigmoid_p(beta_current, xi, y))
            dldb = dldb + correct_vec
        dldb_step = dldb * gamma
        beta_current = do_beta_step(beta_current, gamma, dldb_step)
    return beta_current


def classificator(beta, test_data):
    all1 = 0.0
    error1 = 0.0
    all2 = 0.0
    error2 = 0.0
    x_test = []
    y_test = []
    data_length, vector_len = np.shape(test_data)
    for i in range(data_length):
        x = test_data[i][vector_len-1]
        y = np.dot(beta, test_data[i][:vector_len-1])
        x_test.append(x)
        if y > 0:
            y_test.append(1)
        else:
            y_test.append(-1)
        if x == 1:
            all1 += 1
            if y < 0:
                error1 += 1
        else:
            all2 += 1
            if y >= 0:
                error2 += 1
    print(confusion_matrix(x_test, y_test, labels=[-1, 1]))
    print("Label 1 all test: " + str(all1) + " error: " + str(error1) + ", accuracy: " + str((all1 - error1)/all1)
          + " \nLabel 0 all test : " + str(all2) + " error: " + str(error2) + ", accuracy: " + str((all2 - error2)/all2))


def logistic_regression():
    filename = './datasource/spambase.data'
    arr = load_data(filename)
    arr = np.array(arr)
    data_arr = normalization(arr)
    train_data, test_data = split_arr(data_arr)
    beta0 = intiliaze_beta(train_data)
    gamma = 5 * 10 ** (-3)
    iteration_number = 1000
    beta = find_beta(train_data, beta0, gamma, iteration_number)
    classificator(beta, test_data)
# Label 1 all test: 360.0 error: 48.0, accuracy: 0.866666666667
# Label 0 all test : 560.0 error: 31.0, accuracy: 0.944642857143


def xor_pro():
    train_data = np.array(([0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]))
    beta0 = intiliaze_beta(train_data)
    gamma = 1
    iteration_number = 10
    beta = find_beta(train_data, beta0, gamma, iteration_number)
    classificator(beta, train_data)
    train_data = np.array(([0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]))
    beta0 = intiliaze_beta(train_data)
    beta = find_beta(train_data, beta0, gamma, iteration_number)
    classificator(beta, train_data)
    train_data = np.array(([0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]))
    beta0 = intiliaze_beta(train_data)
    beta = find_beta(train_data, beta0, gamma, iteration_number)
    classificator(beta, train_data)

# logistic_regression()
xor_pro()

