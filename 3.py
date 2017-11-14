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
    

def getCovMatr(data, mean):
    cov = zeros((57,57))
    for vector in data:
        vector.shape=(57,1)
        cov = cov + (vector - mean) * transpose(vector - mean)
    return cov/len(data)
    
def getVariance(data, mean):
    var = zeros((57))
    for vector in data:
        var= var + (vector - mean )**2
    return sqrt(var/len(data))
    
def getFischerLine(klassetruem, klassefalsem, covklassetrue, covklassefalse):
    return matmul((covklassetrue + covklassefalse )** (-1), (klassetruem - klassefalsem))
    
    #sdfs
 #stdDev = varianz           
def getProb(mean, stdDev, arr):
    exponent= exp(-(array(arr) - array(mean))** 2)/(2* (stdDev** 2))
    prob=(1/sqrt(2* pi)* stdDev)* exponent
    return sum(prob)
    

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
    
def klassifikator(test_data, classtruem, classfalsem, vartrue, varfalse):
    all1 = 0
    all2 = 0
    error1 = 0
    error2 = 0
    classtrue, classfalse= getclasses(test_data)
    for vector in classtrue:
        ptrue = getProb(classtruem, vartrue, vector)
        pfalse = getProb(classfalsem, varfalse, vector)
        print(ptrue, pfalse)
        if ptrue >= pfalse:
            all1 += 1
        else:
            all1 +=1
            error1 += 1
    for vector in classfalse:
        ptrue = getProb(classtruem, vartrue, vector)
        pfalse = getProb(classfalse, varfalse, vector)
        print(ptrue, pfalse)
        if pfalse >= ptrue:
            all2 += 1
        else:
            all2 +=1
            error2 += 1
    print("0 :All tests: ", all1)
    print("Count of all errors: ", error1)
    print("Error for every Digit: ", error1/all1)
    print("1 :All tests: ", all2)
    print("Count of all errors: ", error2)
    print("Error for every Digit: ", error2/all2)
    return 0

def gda():
    filename = 'H:/Studium/zweiteSemester/Mustererkennung/Assignment/mustererkennungU1/datasource/spambase.data'
    #filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spamb3ase.data'
    arr = loadData(filename)
    random.shuffle(arr)
    train_data = arr[:int((len(arr) + 1) * .80)]  # Remaining 80% to training set
    test_data = arr[int(len(arr) * .80 + 1):]  # Splits 20% data to test set

    classtrue, classfalse= getclasses(train_data)
    classtruearr = array(classtrue)
    classfalsearr = array(classfalse)
    classtruem = getMean(classtruearr)
    classfalsem = getMean(classfalsearr)
    covclasstrue = getCovMatr(classtruearr, classtruem)
    covclassfalse = getCovMatr(classfalsearr, classfalsem)
    fisheralpha = getFischerLine(classtruem, classfalsem, covclasstrue, covclassfalse)
    classtruem = classtruem * fisheralpha
    classfalsem = classfalsem * fisheralpha
    vartrue = getVariance(classtruearr, classtruem)
    varfalse = getVariance(classfalsearr, classfalsem)
    klassifikator(test_data, classtruem, classfalsem, vartrue, varfalse)
    
    
gda()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    