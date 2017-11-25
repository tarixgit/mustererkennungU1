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
