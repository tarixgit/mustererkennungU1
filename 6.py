from math import *
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import colors


# load the data by one
def loadData(filename):
    file = open(filename, 'r')
    digitarrs = []
    for line in file:
        arr = list(map(float, line.split(',')))
        digitarrs.append(arr)
    file.close()
    return digitarrs

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


def k_means():
    filename = './datasource/iris.data'
    lists = loadData(filename)
    arr = array(lists)
    guetes = []
    for i in range(2, 10):
        clusters = clustering(arr, i)
        guetes.append(guete(clusters))
    poltguete(guetes, clusters)




k_means()
