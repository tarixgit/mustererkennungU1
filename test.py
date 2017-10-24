from numpy import *

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


def n3classifier(testingarr, trainingdict):
    arr = array(testingarr)

    trainingarr = array(trainingdict[9])
    differencearr = trainingarr - arr
    differencearrsquare = differencearr ** 2
    differencearrsum = differencearrsquare.sum(axis=1)
    distancesarr = differencearrsum ** 0.5
    distancesorted = sort(distancesarr)
    minimum1 = distancesorted[0]
    minimum2 = distancesorted[1]
    minimum3 = distancesorted[2]
    index1 = 9
    index2 = 9
    index3 = 9
    for digit in range(9):
        trainingarr = array(trainingdict[digit])
        differencearr = trainingarr - arr
        differencearrsquare = differencearr ** 2
        differencearrsum = differencearrsquare.sum(axis=1)
        distancesarr = differencearrsum ** 0.5
        potentialmin = distancesarr.min()
        if potentialmin < minimum1:
            minimum3 = minimum2
            minimum2 = minimum1
            minimum1 = potentialmin
            index3 = index2
            index2 = index1
            index1 = digit
        else:
            if potentialmin < minimum2:
                minimum3 = minimum2
                minimum2 = potentialmin
                index3 = index2
                index2 = digit
            else:
                if potentialmin < minimum3:
                    minimum3 = potentialmin
                    index3 = digit
    result = index1
    if index2 == index3:
        result = index2
    return result


def n2classifier(testingarr, trainingdict):
    arr = array(testingarr)

    trainingarr = array(trainingdict[9])
    differencearr = trainingarr - arr
    differencearrsquare = differencearr ** 2
    differencearrsum = differencearrsquare.sum(axis=1)
    distancesarr = differencearrsum ** 0.5
    distancesorted = sort(distancesarr)
    minimum1 = distancesorted[0]
    minimum2 = distancesorted[1]
    index1 = 9
    index2 = 9
    for digit in range(9):
        trainingarr = array(trainingdict[digit])
        differencearr = trainingarr - arr
        differencearrsquare = differencearr ** 2
        differencearrsum = differencearrsquare.sum(axis=1)
        distancesarr = differencearrsum ** 0.5
        potentialmin = distancesarr.min()
        if potentialmin < minimum1:
            minimum2 = minimum1
            minimum1 = potentialmin
            index2 = index1
            index1 = digit
        else:
            if potentialmin < minimum2:
                minimum2 = potentialmin
                index2 = digit

    return index2


def n1classifier(testingarr, trainingdict):
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
    errors = 0
    all = 0
    trainingdict = taketrainingdict(trainigfolder)
    errorrate = {}
    for i in range(10):
        errorrate[i] = 0
    for line in testfile:
        digit = int(line[:1])
        #arr = list(map(lambda x: round(float(x) + 1, 5), line[2:].split(' ')))
        arr = list(map(float, line[2:].split(' ')))
        foundDigit = n2classifier(arr, trainingdict)
        if digit != foundDigit:
            errors += 1
            errorrate[digit] += 1
        all += 1
    print("All tests: ", all)
    print("Count of all errors: ", errors)
    print("Error for every Digit: ", errorrate)


read()