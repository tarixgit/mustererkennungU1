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

def initw(ptrain, ntrain):
    arr = array([])
    arr.append(ptrain)
    arr.append(ntrain)
    x = random.randint(0, len(arr)-1)
    w = arr[x]
    return w

def getw(ptrain, ntrain):
    parr = ptrain[:4]
    narr = ntrain[:4]
    wbefore = initw(ptrain, ntrain)
    wafter = [0, 0, 0, 0]
    while(equal(wbefore, wafter)):
        for p in ptrain:
            if dot(w, p) < 0:
                w = sum(w, p, axis=1)
        for n in ntrain:
            if dot(w, n > 0):
                w = sum(w, n , axis=1)
        wafter = w
    return wafter

def perception(ptrain, ntrain, ptest, ntest):
    w = getw(ptrain, ntrain)
    error1 = 0
    error2 = 0
    all1 = 0
    all2 = 0
    for p in ptest:
        all1 += 1
        if dot(w, p) <= 0:
            error1 += 1
    for n in ntest:
        all2 += 1
        if dot(w, n) >= 0:
            error2 += 1
    print("rate1: " + error1/all1 + ", rate2: " + error2/all2)
    return 0

def initalizeClusters(numberofcluster, arr):
    numberOfVector = len(arr) # for random
    clusters = []
    for i in range(numberofcluster):
        r = random.randint(1, numberOfVector)
        mean = arr[r]
        cov = identity(2)
        cluster = Cluster([], cov, mean)
        clusters.append(cluster)
    return clusters

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

def k_means():
    filename = './datasource/iris.data'
    setosa_list, versicolor_list, virginica_list = loadData(filename)
    setosa_versicolor = setosa_list + versicolor_list
    setosa_virginica = setosa_list + virginica_list
    arr_set_ver = array(setosa_versicolor)
    arr_set_vir = array(setosa_virginica)
    train_set_ver, test_set_ver = splitArr(arr_set_ver)
    train_set_vir, test_set_vir = splitArr(arr_set_vir)
    #maybe for future
    #test_set_verP, test_set_verN = getClasses(test_set_ver, 1.0, 2.0)
    train_set_verP, train_set_verN = getClasses(train_set_ver, 1.0, 2.0)
    filename = './datasource/iris.data'





k_means()
