from numpy import *
from os import listdir
from os.path import isfile, join
from os import walk

def fib(n):    # die Fibonacci-Folge bis n ausgeben
    # """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b


def start():
    fib(2000)


def test():
    f = []
    for dirpath, dirnames, filenames in walk('C:/Users/Taras/mustererkennungU1/datasource/training'):
        f.extend(filenames)
        print(filenames)
        break
    file = 'C:/Users/Taras/mustererkennungU1/datasource/training/' + filenames[0]
    f = open(file, 'r')
    print(f.readline())


def taketrainingdict(trainigfolder ):
    output = {}
    for digit in range(10):
        filename = trainigfolder + 'train.' + str(digit)
        file = open(filename, 'r')
        digitarrs = []
        for line in file:
            arr = list(map(float, line.split(',')))
            digitarrs.append(arr)
        output[digit] = digitarrs
    return output


def nnclassifier(testingarr, trainingdict):
    arr = array(testingarr)

    trainingarr = array(trainingdict[9])
    differencearr = trainingarr - arr
    differencearrsquare = differencearr ** 2
    differencearrsum = differencearrsquare.sum(axis=1)
    distancesarr = differencearrsum ** 0.5
    minimum = distancesarr.min()
    index = 9
    for digit in range(9):
        trainingarr = array(trainingdict[digit])
        differencearr = trainingarr - arr
        differencearrsquare = differencearr ** 2
        differencearrsum = differencearrsquare.sum(axis=1)
        distancesarr = differencearrsum ** 0.5
        potentialmin = distancesarr.min()
        if potentialmin < minimum:
            minimum = potentialmin
            index = digit
    return index


def read():
    testfilename = 'C:/Users/Taras/mustererkennungU1/datasource/test/zip.test'
    trainigfolder = 'C:/Users/Taras/mustererkennungU1/datasource/training/'
    testfile = open(testfilename, 'r')
    print("File for testing is open: ", testfile.name)
    s = 0
    all = 0
    for line in testfile:
        digit = int(line[:1])
        #arr = list(map(lambda x: round(float(x) + 1, 5), line[2:].split(' ')))
        arr = list(map(float, line[2:].split(' ')))
        trainingdict = taketrainingdict(trainigfolder)
        foundDigit = nnclassifier(arr, trainingdict)
        if digit == foundDigit:
            s += 1
        all += 1
    print(s)
    print(all)


read()