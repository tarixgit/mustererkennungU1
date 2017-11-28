from math import *
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import colors


# load the data by one
def loadData(filename):
    file = open(filename, 'r')
    digitarrs = []
    # Iris-setosa == 1
    # Irisversicolor == 2
    # Iris-virginica == 3
    setosa_arr = []
    versicolor_arr = []
    virginica_arr = []
    for line in file:
        listofattr  = line.split(',')
        endindex = len(listofattr)-1
        if listofattr[endindex].strip() == 'Iris-setosa':
            listofattr[endindex] = '1'
            setosa_arr.append(list(map(float, listofattr)))
        if listofattr[endindex].strip() == 'Iris-versicolor':
            listofattr[endindex] = '2'
            versicolor_arr.append(list(map(float, listofattr)))
        if listofattr[endindex].strip() == 'Iris-virginica':
            listofattr[endindex] = '3'
            virginica_arr.append(list(map(float, listofattr)))
    file.close()
    return setosa_arr, versicolor_arr, virginica_arr


def splitArr(arr):
    random.shuffle(arr)
    train_data = arr[:int((len(arr) + 1) * .80)]  # Remaining 80% to training set
    test_data = arr[int(len(arr) * .80 + 1):]  # Splits 20% data to test set
    return train_data, test_data


def getClasses(arr, p_mark, n_mark):
    vectorlen = len(arr[0])
    resulttrue = []
    resultfalse = []
    for vector in arr:
        if vector[vectorlen-1] == p_mark:
            resulttrue.append(vector[:vectorlen-1])
        if vector[vectorlen-1] == n_mark:
            resultfalse.append(vector[:vectorlen-1])
    return resulttrue, resultfalse


def initw(train_data):
    x = random.randint(0, len(train_data)-1)
    w = train_data[x]
    return w[:len(w)-1]


def isPositive(vector, p_mark, n_mark):
    vectorlen = len(vector)
    if vector[vectorlen - 1] == p_mark:
        return True
    if vector[vectorlen - 1] == n_mark:
        return False


def isMultNegative(vector, w):
    return dot(vector, w) < 0


def getw(train_data, p_mark, n_mark):
    wbefore = initw(train_data) # w-zero
    wafter = array([0, 0, 0, 0])
    t = 0
    while(not array_equal(wbefore, wafter)):
        if not t == 0:
            wbefore = wafter
        for vector in train_data:
            vector_short = vector[:len(vector)-1]
            if isPositive(vector, p_mark, n_mark) and isMultNegative(vector_short, wbefore):
                wafter = wbefore + vector_short
                t += 1
                pass
            elif not (isPositive(vector, p_mark, n_mark)) and not (isMultNegative(vector_short, wbefore)):
                wafter = wbefore - vector_short
                t += 1
                pass
    print(t)
    return wafter


def bestw(train_data, itr, p_mark, n_mark):
    w_before = initw(train_data)
    w_after = array([0, 0, 0, 0])
    w_best = array([0, 0, 0, 0])
    p, n = getClasses(train_data, p_mark, n_mark)
    p_len = len(p)
    n_len = len(n)
    h_before = p_len + n_len
    for i in range(itr):
        for vector in train_data:
            h_p = 0
            h_n = 0
            vector_short = vector[:len(vector)-1]
            if isPositive(vector, p_mark, n_mark) and (dot(vector_short, w_before) < 0):
                w_after = w_before + vector_short
            elif not (isPositive(vector, p_mark, n_mark)) and (dot(vector_short, w_before) >= 0):
                w_after = w_before - vector_short
            if not (w_after == w_before).all():
                for p_vector in p:
                    if dot(w_after, p_vector) >= 0:
                        h_p += 1
                for n_vector in n:
                    if dot(w_after, n_vector) < 0:
                        h_n += 1
                h_p = (p_len - h_p) + (n_len - h_n)
                if h_p < h_before:
                    w_best = w_after
                    h_before = h_p
                    print(h_p)
            w_before = w_after
    return w_best


def errorRate(ptest, ntest, w):
    error1 = 0.0
    error2 = 0.0
    all1 = 0.0
    all2 = 0.0
    for p in ptest:
        all1 += 1
        if dot(w, p) < 0:
            error1 += 1
    for n in ntest:
        all2 += 1
        if dot(w, n) >= 0:
            error2 += 1
    print("All1: " + str(all1) + " error1: " + str(error1) + ", All2: " + str(all2) + " error2: " + str(error2))
    return 0


def perception():
    filename = './datasource/iris.data'
    setosa_list, versicolor_list, virginica_list = loadData(filename)
    setosa_versicolor = setosa_list + versicolor_list
    setosa_virginica = setosa_list + virginica_list
    versicolor_virgiica = versicolor_list + virginica_list
    plt.figure(figsize=(15, 10))
    plt.plot(versicolor_list, virginica_list)
    arr_set_ver = array(setosa_versicolor)
    arr_set_vir = array(setosa_virginica)
    arr_ver_vir = array(versicolor_virgiica)

    # train_set_ver, test_set_ver = splitArr(arr_set_ver)
    # test_set_verP, test_set_verN = getClasses(test_set_ver, 1.0, 2.0)
    # errorRate(test_set_verP, test_set_verN, getw(train_set_ver, 1.0, 2.0))
    #
    # train_set_vir, test_set_vir = splitArr(arr_set_vir)
    # test_set_virP, test_set_virN = getClasses(test_set_vir, 1.0, 3.0)
    # errorRate(train_set_vir, test_set_virP, getw(train_set_ver, 1.0, 3.0))

    itr = 1000
    train_ver_vir, test_ver_vir = splitArr(arr_ver_vir)
    test_ver_virP, test_ver_virN = getClasses(test_ver_vir, 2, 3)
    errorRate(test_ver_virP, test_ver_virN, bestw(train_ver_vir, itr, 2, 3))




perception()
