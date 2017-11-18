from math import *
from numpy import *
import matplotlib.pyplot as plt


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

def splittedInCluster(arr, covariances, means):
    numberofcluster = len(means)
    for vector in arr:
        distances = array([])
        for i in len(mean):
            cov = covariances[i]
            mean = means[i]
            distances.append(distance(vector, mean, cov))
        clusterAssignIndex = argmin(distances)
        clusters[clusterAssignIndex].append(vector)

    return


def getFirstCovariances(numberofcluster):
    x = array([])
    for i in range(numberofcluster):
        x.append(identity())
    return x

def clustering(arr):
    len = len(arr)
    for numberofcluster  in range(2, 10):

        means = getFirstMeans(numberofcluster, arr)
        covariances = getFirstCovariances()
        splittedClusterData = splittedInCluster(arr, covariances, means)

        for i in range(numberofcluster):
            x = random.randint(1, len)
            means.append(arr[x])

            covariance = identity()
            findmean()

def guete(covArr):
    guete[]
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














