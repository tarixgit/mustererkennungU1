from math import *
from numpy import *
import matplotlib.pyplot as plt

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

def getProjMean(arr, fisheralpha):
    sum = 0
    for vector in arr:
        num = getProjection(vector, fisheralpha)
        sum += num
    return sum/len(arr)

def getProjection(vektor, alpha):
    return matmul(vektor, alpha)

def getCovMatr(data, mean):
    cov = zeros((57,57))
    for vector in data:
        vector.shape=(57,1)
        cov = cov + (vector - mean) * transpose(vector - mean)
    return cov/len(data)

def getVariance(data, mean, alpha):
    #var = zeros((57))
    var = 0
    for vector in data:
        var= var + (getProjection(vector, alpha) - mean )**2
    return sqrt(var/len(data))
    
def getFischerLine(klassetruem, klassefalsem, covklassetrue, covklassefalse):
    return matmul((covklassetrue + covklassefalse )** (-1), (klassetruem - klassefalsem))
    
    #sdfs
 #stdDev = varianz           
def getProb(mean, stdDev, arr):
    exponent = exp(-((arr - mean) ** 2) / (2 * (stdDev ** 2)))
    prob = (1 / (sqrt(2 * pi) * stdDev)) * exponent
    return prob
    

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
    
def klassifikator(test_data, classtruem, classfalsem, vartrue, varfalse, fisheralpha):
    all1 = 0
    all2 = 0
    error1 = 0
    error2 = 0
    classtrue, classfalse= getclasses(test_data)
    x = []
    y = []
    x2 = []
    y2 = []
    for vector in classtrue:
        projfvector = getProjection(vector, fisheralpha)
        ptrue = getProb(classtruem, vartrue, projfvector)
        pfalse = getProb(classfalsem, varfalse, projfvector)
        print(ptrue, pfalse)
        if ptrue >= pfalse:
            all1 += 1
        else:
            all1 +=1
            error1 += 1
        y.append(ptrue)
        x.append(projfvector)
    for vector in classfalse:
        projfvector = getProjection(vector, fisheralpha)
        ptrue = getProb(classtruem, vartrue, projfvector)
        pfalse = getProb(classfalsem, varfalse, projfvector)
        print(ptrue, pfalse)
        if pfalse >= ptrue:
            all2 += 1
        else:
            all2 +=1
            error2 += 1
        y2.append(pfalse)
        x2.append(projfvector)
    print("0 :All tests: ", all1)
    print("Count of all errors: ", error1)
    print("Error for every Digit: ", error1/all1)
    print("Success: ", (1 - error1 / all1))
    print("1 :All tests: ", all2)
    print("Count of all errors: ", error2)
    print("Error for every Digit: ", error2/all2)
    print("Success: ", (1 - error2/all2))
    plt.plot(x, y, 'r+', x2, y2, 'g+')
    plt.show()
    return 0

def plotFunctions(classtrue, classflase, alpha):
    resarr1 = []
    resarr2 = []
    for vector in classtrue:
        resarr1.append(getProjection(vector, alpha))

    for vector in classflase:
        resarr2.append(getProjection(vector, alpha))

    return resarr1, resarr2

def gda():
    #filename = 'H:/Studium/zweiteSemester/Mustererkennung/Assignment/mustererkennungU1/datasource/spambase.data'
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

    covclasstrue = cov(transpose(classtruearr))
    covclassfalse = cov(transpose(classfalsearr))

    fisheralpha = getFischerLine(classtruem, classfalsem, covclasstrue, covclassfalse)

    #classtruemP = getProjMean(classtruearr, fisheralpha)
    #classfalsemP = getProjMean(classfalsearr, fisheralpha)

    classtruem = getProjection(classtruem, fisheralpha)
    classfalsem = getProjection(classfalsem, fisheralpha)
    vartrue = getVariance(classtruearr, classtruem, fisheralpha)
    varfalse = getVariance(classfalsearr, classfalsem, fisheralpha)
    klassifikator(test_data, classtruem, classfalsem, vartrue, varfalse, fisheralpha)
    a, b = plotFunctions(classtruearr, classfalsearr, fisheralpha)
    #plt.plot(a, , color="green", b, , color="green")
    plt.plot(a)
    #plt.plot.acorr()
    plt.show()
    
gda()

#import sklearn
#train_test_split
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    