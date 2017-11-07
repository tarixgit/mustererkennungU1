from numpy import *

#load the training data by one
def loadTrain(trainigfolder,digit ):
    filename = trainigfolder + 'train.' + str(digit)
    file = open(filename, 'r')
    m= len(file.readlines())
    dataArray = ones((m,257))
    arr = zeros((1,256))
    file = open(filename, 'r')
    for i in range(m):
        arr = list(map(float, file.readline().split(',')))
        for j in range(256):
            dataArray[i,j+1]= arr[j]
    return dataArray

#load the training data of two digits,
#set y as 1 for the first digit,
#set y as -1 for the second digit   
def loadData(trainigfolder, digit1, digit2):
    dataArray1= loadTrain(trainigfolder, digit1)
    dataArray2= loadTrain(trainigfolder, digit2)
    m= len(dataArray1)
    n= len(dataArray2)
    dataArray= zeros((m+n,257))
    labelArray= zeros((m+n,1))
    for i in range(m):
        dataArray[i,:]= dataArray1[i,:]
        labelArray[i,0]= 1
    for j in range(n):
        dataArray[m+j,:]= dataArray2[j,:]
        labelArray[m+j,0]= -1
    return dataArray, labelArray


# def getPhi(dataArray, labelArray):
#     phi=(dataArray*transpose(dataArray))**(-1)*transpose(dataArray)*labelArray
#     return phi

#iterator to get a better m,b accorading to the learningRate and numIter  
def optimizer(m, b, dataArray, labelArray, learningRate, numIter):
    for i in range(numIter):
        m,b= getGradient(m, b, dataArray, labelArray, learningRate)
    return m, b

#use Gradient descent to get m,b    
def getGradient(m, b, dataArray, labelArray, learningRate):
    bGradient= 0
    mGradient= zeros(257)
    length= len(labelArray)
    for i in range(length):
        bGradient+= -(2/float(length))*(labelArray[i,0]-(sum(m* dataArray[i,:])+b))
        mGradient+= -(2/float(length))*dataArray[i,:]*(labelArray[i,0]-(sum(m* dataArray[i,:])+b))
    bNew= b-(learningRate * bGradient)
    mNew= m-(learningRate * mGradient)
    return mNew, bNew
    
def klassifikator(testfilename, trainigfolder,digit1, digit2, m, b, learningRate, numIter): 
    dataArray, labelArray= loadData(trainigfolder, digit1, digit2)
    m,b= optimizer(m, b, dataArray, labelArray, learningRate, numIter)
    testfile = open(testfilename, 'r')
    errors1= 0.0
    errors2= 0.0
    all1= 0.0
    all2= 0.0
    for line in testfile:
        digit = int(line[:1])
        if digit== digit1 or digit== digit2:
            arr = list(map(float, line[2:].split(' ')))
            arr.insert(0,1)
            result=sum(m*arr)+b
            if result >= 0 :
                foundDigit=digit1
            else:
                foundDigit=digit2
            if digit== digit1:
                if digit != foundDigit:
                    errors1 += 1
                all1 += 1
            if digit== digit2:
                if digit != foundDigit:
                    errors2 += 1
                all2 +=1
    print(digit1,":All tests: ", all1)
    print("Count of all errors: ", errors1)
    print("Error for every Digit: ", errors1/all1)
    print(digit2,":All tests: ", all2)
    print("Count of all errors: ", errors2)
    print("Error for every Digit: ", errors2/all2)
      
        
        
def linearRegression():
    testfilename = 'H:/Studium/zweiteSemester/Mustererkennung/Assignment/test/zip.test'
    trainigfolder = 'H:/Studium/zweiteSemester/Mustererkennung/Assignment/training/'
    b= 0
    m= zeros(257)
    learningRate= 0.0001
    numIter= 1000 
    klassifikator(testfilename, trainigfolder,3, 5, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder,3, 7, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder,3, 9, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder,5, 7, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder,5, 9, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder,7, 9, m, b, learningRate, numIter)
    
    
    
linearRegression()


    

    