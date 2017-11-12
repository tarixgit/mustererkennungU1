def fisher():
    filename = '/home/tarix/PycharmProjects/mustererkennungU1/datasource/spambase.data'
    file = open(filename, 'r')

    digitarrs = []
    for line in file:
        arr = list(map(float, line.split(',')))
        digitarrs.append(arr)
    output[digit] = digitarrs



    klassifikator(testfilename, trainigfolder, 3, 5, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder, 3, 7, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder, 3, 9, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder, 5, 7, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder, 5, 9, m, b, learningRate, numIter)
    klassifikator(testfilename, trainigfolder, 7, 9, m, b, learningRate, numIter)


linearRegression()
