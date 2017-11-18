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

def clustering(arr):
    len = len(arr)
    for numberofcluster in range(10):

        means = getFirstMeans(numberofcluster, arr)
        covariance = identity()

        for i in range(numberofcluster):
            x = random.randint(1, len)
            means.append(arr[x])

            covariance = identity()
            findmean()

def distance(vektor, mean, cov):
    distance = (vektor - mean) * (1 / cov) * (vektor - mean)
    return distance

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














