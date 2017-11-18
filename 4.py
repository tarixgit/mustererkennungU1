from math import *
from numpy import *
import matplotlib.pyplot as plt


class Cluster:

    #kind = 'canine'         # class variable shared by all instances

    #covariance = cov
    def __init__(self, arr, cov, center):
        self.arr = arr
        self.cov = cov
        self.center = center

# load the data by one
def loadData(filename):
    file = open(filename, 'r')
    digitarrs = []
    for line in file:
        arr = list(map(float, line.split(',')))
        digitarrs.append(arr)
    file.close()
    return digitarrs

def getProb(mean, stdDev, arr):
    exponent = exp(-((arr - mean) ** 2) / (2 * (stdDev ** 2)))
    prob = (1 / (sqrt(2 * pi) * stdDev)) * exponent
    return prob

#not used more
def getFirstMeans(numberofcluster, arr):
    numberOfVector = len(arr)
    means = array([])
    for i in range(numberofcluster):
        x = random.randint(1, numberOfVector)
        means.append(arr[x])
    return means

def distance(vektor, mean, cov):
    distance = (vektor - mean) * (1 / cov) * (vektor - mean)
    return distance

def splittedInCluster(arr, clusters):
    numberofcluster = len(clusters)
    vectorlen = len(arr[0])
    clusters = [] #list of cluster objects
    # if means_before != means:
    for vector in arr:
        distances = array([])
        for i in len(mean):
            cov = covariances[i]
            mean = means[i]
            distances.append(distance(vector, mean, cov))
        clusterAssignIndex = argmin(distances)
        clusters[clusterAssignIndex].append(vector)

    return clusters;

#not used more
def getFirstCovariances(numberofcluster):
    x = array([])
    for i in range(numberofcluster):
        x.append(identity())
    return x

def initalizeClusters(numberofcluster, arr):
    numberOfVector = len(arr) # for random
    clusters = []
    for i in range(numberofcluster):
        r = random.randint(1, numberOfVector)
        mean = arr[r]
        cov = identity()
        cluster = Cluster(array([]), cov, mean)
        clusters.append(cluster)
    return clusters

def clustering(arr):
    len = len(arr)
    #maybe to change the loop to elbogen method, where you dont stop until you broder (x/y == 1) overcomme
    for numberofcluster  in range(2, 10):
        # fuer jeder cluster
        # 1) cluster finded
        # 2) covarianz bekommen und Guete berechnen


        clusters = initalizeClusters(numberofcluster, arr)

        clusters = splittedInCluster(arr, clusters) #renew


def k_means():
    # filename = 'H:/Studium/zweiteSemester/Mustererkennung/Assignment/mustererkennungU1/datasource/spambase.data'
    #filename = '~/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    filename = './datasource/2d-em.csv'
    lists = loadData(filename)
    arr = array(lists)
    clustering(arr)

    #plt.plot(a)
    #plt.show()


k_means()

# import sklearn
# train_test_split














