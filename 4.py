from math import *
from numpy import *
import matplotlib.pyplot as plt


class Cluster:

    #cov = 'canine'         # class variable shared by all instances

    #covariance = cov
    def __init__(self, arr, cov, mean):
        self.arr = arr
        self.cov = cov      #old covevrgance, from cluster before
        self.mean = mean

# load the data by one
def loadData(filename):
    file = open(filename, 'r')
    digitarrs = []
    for line in file:
        arr = list(map(float, line.split(',')))
        digitarrs.append(arr)
    file.close()
    return digitarrs

#not used
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

def distance(vektor, center, cov):
    dif = vektor - center
    #dif.shape = (len(dif), 1)
    if array_equal(cov, identity(2)):
        mult = dif
    else:
        mult = matmul(dif, cov ** (-1))
    distance = matmul(mult, (vektor - center))
    return distance

def checkifthesame(clusters):
    same = True
    #what if some cluster doesn't have points(arr is empty)?
    for cluster in clusters:
        old_mean = cluster.mean
        cluster.mean = mean(cluster.arr) #check if this work, maybe wrong achse
        cluster.cov = cov(cluster.arr)  ## must I to transpose????
        if cluster.mean != old_mean:   ## to change with threshold, i suppose it must be not directly same
            same = False
            break
    return same

def splittedInCluster(arr, clusters):
    numberofcluster = len(clusters)
    # if means_before != means:
    same = True
    while same:
        for vector in arr:
            distances = []
            for i in range(numberofcluster):
                distances.append(distance(vector, clusters[i].mean, clusters[i].cov))
            cluster_assign_index = argmin(distances)
            if clusters[cluster_assign_index].arr == []:
                clusters[cluster_assign_index].arr = array(vector, ndmin=2)
            else:
                clusters[cluster_assign_index].arr.append(vector)
        same = checkifthesame(clusters)
        notsame = not same
#to test
    return clusters

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
        cov = identity(2)
        cluster = Cluster([], cov, mean)
        clusters.append(cluster)
    return clusters

def clustering(arr):
    length = len(arr)
    #maybe to change the loop to elbogen method, where you dont stop until you broder (x/y == 1) overcomme
    for numberofcluster  in range(2, 10):
        # fuer jeder cluster
        # 1) cluster finded
        # 2) covarianz bekommen und Guete berechnen


        clusters = initalizeClusters(numberofcluster, arr)
        clustersnew = splittedInCluster(arr, clusters) #renew


def guete(covArr):
    for i in range(covArr.shape[0]):
        guete.append(linalg.norm(covArr[i]))
        #guete.append(linalg.norm(covArr[i], inf))
    return guete

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














