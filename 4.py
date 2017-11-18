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




def getProjection(vektor, alpha):
    return matmul(vektor, alpha)

def getVariance(data, mean, alpha):
    # var = zeros((57))
    var = 0
    for vector in data:
        var = var + (getProjection(vector, alpha) - mean) ** 2
    return sqrt(var / len(data))


def getFischerLine(klassetruem, klassefalsem, covklassetrue, covklassefalse):
    return matmul((covklassetrue + covklassefalse) ** (-1), (klassetruem - klassefalsem))

    # sdfs
    # stdDev = varianz


def getProb(mean, stdDev, arr):
    exponent = exp(-((arr - mean) ** 2) / (2 * (stdDev ** 2)))
    prob = (1 / (sqrt(2 * pi) * stdDev)) * exponent
    return prob

def gda():
    # filename = 'H:/Studium/zweiteSemester/Mustererkennung/Assignment/mustererkennungU1/datasource/spambase.data'
    filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    arr = loadData(filename)


    #plt.plot(a)
    #plt.show()


gda()

# import sklearn
# train_test_split














