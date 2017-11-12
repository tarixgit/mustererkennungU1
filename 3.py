from math import *
from numpy import *

#load the data by one
def loadData(filename):
    file = open(filename, 'r')
    digitarrs = []
    for line in file:
        arr = list(map(float, line.split(',')))
        digitarrs.append(arr)
    file.close()
    return digitarrs
    
def getMean(arr):
    mean= sum(arr)/float(len(digitarrs))
    return mean
    
def getSstdDev(digitarrs):
    stdDev= sqrt(sum((digitarrs-getMean(digitarrs))** 2))/float(len(digitarrs))
    return stdDev
# def getStdDev(digitarrs):
#     stdDev= sqrt(sum((digitarrs-getMean(digitarrs))** 2))
#     return stdDev

def getCovMatr(data, mean):
    cov = [[]]
    for i in len(data):
        cov = cov + (data[i] - mean)*(transpose(data[i] - mean))
    return cov/len(data)
    
def getFischerLine(klassetruem, klassefalsem, covklassetrue, covklassefalse):
    return (covklassetrue + covklassefalse )** (-1)*( klasseturem- klassefalsem)  
    
    #sdfs
 #stdDev = varianz           
def getProb(mean, stdDev, arr):
    exponent= exp(-(arr- mean)** 2)/(2* (stdDev** 2))
    prob=(1/sqrt(2* pi)* stdDev)* exponent
    return prob
    
def getMS(dataArrays, digit):
    
    return 0
    
def klassifikator(testfilename, trainigfolder, digit1, digit2):
    digitarrs1= loadTrain(trainigfolder, digit1)
    digitarrs2= loadTrain(trainigfolder, digit2)
    mean1=getMean(digitarrs1)
    mean2=getMean(digitarrs2)
    stdDev1=getStdDev(digitarrs1)
    stdDev2=getStdDev(digitarrs2)
    print(mean1,stdDev1)
    testfile = open(testfilename, 'r')
    errors1= 0.0
    errors2= 0.0
    all1= 0.0
    all2= 0.0
    for line in testfile:
        digit = int(line[:1])
        if digit== digit1 or digit== digit2:
            arr = list(map(float, line[2:].split(' ')))
            prob1= getProb(mean1, stdDev1, arr)
            prob2= getProb(mean2, stdDev2, arr)
            print(prob1,prob2)
            pre= max(prob1, prob2)
            if digit== digit1 and pre== prob2:
                errors1+= 1
            all1+= 1
            if digit== digit2 and pre== prob1:
                errors2+= 1
            all2+= 1
    print(digit1,":All tests: ", all1)
    print("Count of all errors: ", errors1)
    print("Error for every Digit: ", errors1/all1)
    print(digit2,":All tests: ", all2)
    print("Count of all errors: ", errors2)
    print("Error for every Digit: ", errors2/all2)
    
def gda():
    filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    arr = loadData(filename)
    np.random.shuffle(arr)
    train_data = data[:int((len(data) + 1) * .80)]  # Remaining 80% to training set
    test_data = data[int(len(data) * .80 + 1):]  # Splits 20% data to test set

    klassetrue =
    klassefalse =

    klassetruem = getMean(klassetrue)
    klassefalsem = getMean(klassetrue)
    covklassetrue = getCovMatr(klassetrue, klassetruem)
    covklassefalse = getCovMatr(klassefalse, klassefalsem)

    klassifikator(testfilename, trainigfolder, 3, 5)
    
    
gda()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    