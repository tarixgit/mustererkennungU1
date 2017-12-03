import numpy as np


# load the data by classify
def load_data(filename):
    file = open(filename, 'r')
    # Iris-setosa == 1
    # Irisversicolor == 2
    # Iris-virginica == 3
    setosa_arr = []
    versicolor_arr = []
    virginica_arr = []
    for line in file:
        listofattr = line.split(',')
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


# separate training data and rest data (80/20%)
# randomize training data list
def split_arr(arr):
    np.random.shuffle(arr)
    train_data = arr[:int((len(arr) + 1) * .80)]  # Remaining 80% to training set
    test_data = arr[int(len(arr) * .80 + 1):]  # Splits 20% data to test set
    return train_data, test_data

# remove label from each vector
def get_classes(arr, p_mark, n_mark):
    vectorlen = len(arr[0])
    resulttrue = []
    resultfalse = []
    for vector in arr:
        if vector[vectorlen-1] == p_mark:
            resulttrue.append(vector[:vectorlen-1])
        if vector[vectorlen-1] == n_mark:
            resultfalse.append(vector[:vectorlen-1])
    return resulttrue, resultfalse

# get initial w0
def initw(train_data):
    x = np.random.randint(0, len(train_data)-1)
    w = train_data[x]
    return w[:len(w)-1]

# whether the vektor is positive
def is_positive(vector, p_mark, n_mark):
    vectorlen = len(vector)
    if vector[vectorlen - 1] == p_mark:
        return True
    if vector[vectorlen - 1] == n_mark:
        return False

# get w
def getw(train_data, p_mark, n_mark):
    # w-zero
    wbefore = initw(train_data)
    wafter = np.array([0, 0, 0, 0])
    t = 0
    while(not np.array_equal(wbefore, wafter)):
        if not t == 0:
            wbefore = wafter
        for vector in train_data:
            vector_short = vector[:len(vector)-1]
            if is_positive(vector, p_mark, n_mark) and np.dot(vector_short, wbefore) < 0:
                wafter = wbefore + vector_short
                t += 1
                pass
            elif not (is_positive(vector, p_mark, n_mark)) and not (np.dot(vector_short, wbefore) < 0):
                wafter = wbefore - vector_short
                t += 1
                pass
    return wafter


# get best w
# define best w: the w which can distinguish the largest number among positive and negative vectors
def bestw(train_data, itr, p_mark, n_mark):
    w_before = initw(train_data)
    w_after = np.array([0, 0, 0, 0])
    w_best = np.array([0, 0, 0, 0])
    p, n = get_classes(train_data, p_mark, n_mark)
    p_len = len(p)
    n_len = len(n)
    h_before = p_len + n_len
    for i in range(itr):
        for vector in train_data:
            h_p = 0
            h_n = 0
            vector_short = vector[:len(vector)-1]
            if is_positive(vector, p_mark, n_mark) and (np.dot(vector_short, w_before) < 0):
                w_after = w_before + vector_short
            elif not (is_positive(vector, p_mark, n_mark)) and (np.dot(vector_short, w_before) >= 0):
                w_after = w_before - vector_short
            if not (w_after == w_before).all():
                for p_vector in p:
                    if np.dot(w_after, p_vector) >= 0:
                        h_p += 1
                for n_vector in n:
                    if np.dot(w_after, n_vector) < 0:
                        h_n += 1
                h_p = (p_len - h_p) + (n_len - h_n)
                if h_p < h_before:
                    w_best = w_after
                    h_before = h_p
            w_before = w_after
    return w_best


def error_rate(ptest, ntest, w):
    error1 = 0.0
    error2 = 0.0
    all1 = 0.0
    all2 = 0.0
    for p in ptest:
        all1 += 1
        if np.dot(w, p) < 0:
            error1 += 1
    for n in ntest:
        all2 += 1
        if np.dot(w, n) >= 0:
            error2 += 1
    print("Cluster 1, all vectors: " + str(all1) + " error 1: " + str(error1) + ", Cluster 2, all vectors: " + str(all2) + " error2: " + str(error2))
    return 0


def perception():
    filename = './datasource/iris.data'
    setosa_list, versicolor_list, virginica_list = load_data(filename)
    setosa_versicolor = setosa_list + versicolor_list
    setosa_virginica = setosa_list + virginica_list
    versicolor_virgiica = versicolor_list + virginica_list
    arr_set_ver = np.array(setosa_versicolor)
    arr_set_vir = np.array(setosa_virginica)
    arr_ver_vir = np.array(versicolor_virgiica)
# Klassifikator: Iris-setosa und Iris-versicolor
    train_set_ver, test_set_ver = split_arr(arr_set_ver)
    test_set_verP, test_set_verN = get_classes(test_set_ver, 1.0, 2.0)
    print("Klassifikator: Iris-setosa und Iris-versicolor")
    error_rate(test_set_verP, test_set_verN, getw(train_set_ver, 1.0, 2.0))
# Klassifikator: Iris-setosa und Iris-virginica
    train_set_vir, test_set_vir = split_arr(arr_set_vir)
    test_set_virP, test_set_virN = get_classes(test_set_vir, 1.0, 3.0)
    print("Klassifikator: Iris-setosa und Iris-virginica")
    error_rate(test_set_virP, test_set_virN, getw(train_set_vir, 1.0, 3.0))
# beste Gewichtsvektor
    itr = 1000
    train_ver_vir, test_ver_vir = split_arr(arr_ver_vir)
    test_ver_virP, test_ver_virN = get_classes(test_ver_vir, 2, 3)
    print("Klassifikator mit bestem Gewichtsvektor: Iris-versicolor und Iris-virginica")
    error_rate(test_ver_virP, test_ver_virN, bestw(train_ver_vir, itr, 2, 3))


perception()
