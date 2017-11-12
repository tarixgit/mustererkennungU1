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
    sumvector = arr.sum(axis=0)
    mean= sumvector/float(len(arr))
    return mean
    
def getSstdDev(digitarrs):
    stdDev= sqrt(sum((digitarrs-getMean(digitarrs))** 2))/float(len(digitarrs))
    return stdDev
# def getStdDev(digitarrs):
#     stdDev= sqrt(sum((digitarrs-getMean(digitarrs))** 2))
#     return stdDev

def getCovMatr(data, mean):
    cov = [[]]
    for vector in data:
        cov = cov + (vector - mean) * transpose(vector - mean)
    return cov/len(data)
    
    
    
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

def getclasses(train_data):
    vectorlen = len(train_data[0])
    resulttrue = []
    resultfalse = []
    for vector in train_data:
        if vector[vectorlen-1] == 1.0:
            resulttrue.append(vector[:vectorlen-1])
        if vector[vectorlen-1] == 0.0:
            resultfalse.append(vector[:vectorlen-1])
    return resulttrue, resultfalse

def gda():
    filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    arr = loadData(filename)
    random.shuffle(arr)
    train_data = arr[:int((len(arr) + 1) * .80)]  # Remaining 80% to training set
    test_data = arr[int(len(arr) * .80 + 1):]  # Splits 20% data to test set

    classtrue, classfalse= getclasses(train_data)
    classtruearr = array(classtrue)
    classfalsearr = array(classfalse)

    classtruem = getMean(classtruearr)
    classfalsem = getMean(classfalsearr)

    #covclasstrue = getCovMatr(classtruearr, classtruem)
    #covclassfalse = getCovMatr(classfalsearr, classfalsem)

    covclasstrue = cov(classtruearr)
    covclassfalse = cov(classfalsearr)



    klassifikator(testfilename, trainigfolder, 3, 5)
    
    
gda()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    