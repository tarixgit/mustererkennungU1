import numpy as np
import matplotlib.pyplot as plt


# achtung
# achtung
# achtung
# achtung
# vielleicht habe ich alle multiplikation falsche gemacht, die genaurichkeit ist ja so immer 0
# achtung
# achtung
# achtung
# achtung
class Data:
    def __init__(self):
        self.traingdata = {}

    def taketrainingdict(self, trainigfolder):
        for digit in range(10):
            filename = trainigfolder + 'train.' + str(digit)
            file = open(filename, 'r')
            digitarrs = []
            for line in file:
                arr = list(map(float, line.split(',')))
                digitarrs.append(arr)
            self.traingdata[digit] = digitarrs


class Layer:
    def __init__(self, neuronen, input_anzahl):  # fuer die esrte Schichte sind es attributen
        self.gewichte = np.random.rand(neuronen, input_anzahl+1)
        self.neuronen = neuronen
        self.input_anzahl = input_anzahl
        self.func = np.ones(neuronen+1)
        self.ableitung = []
        self.e = []
        self.delta = []

    # def calc_func_ableitung(self, input):   # input ist hier vorherige Ergebnisse
    #     input.append(1)
    #     sigmoid = lambda x: 1 / (1 + np.e ** (-x))
    #     vf = np.vectorize(sigmoid)
    #     self.func = vf(input * self.gewichte)    # output auf jeder Schichte       # schritt 1,2 in der Vorlesung
    #     self.ableitung = self.func * (1 - self.func)    # schritt 4,5 in der Vorlesung

    def calc_func_ableitung(self, input):   # input ist hier vorherige Ergebnisse
        sigmoid = lambda x: 1 / (1 + np.e ** (-x))
        s = lambda x, y: x*y
        vf = np.vectorize(sigmoid)
        vs = np.vectorize(s)
        a = np.sum(vs(input, self.gewichte), axis=1)
        for i in range(len(a)):
            self.func[i] = (1 / (1 + np.e ** (- a[i])))   # schritt 1,2 in der Vorlesung
        self.ableitung = MathTools.list_each_list_multiply(self.func, (MathTools.value_list_add(-1, self.func)))    # schritt 4,5 in der Vorlesung

    def calc_e(self, target):      # schritt 3 in der Vorlesung
        targetarr = np.zeros(10)
        targetarr[target] = 1
        self.e = list(map(lambda x: x[0]-x[1], zip(self.func, targetarr)))


class MathTools:
    def __init__(self):
        self = self

    @staticmethod
    def list_each_list_multiply(list1, list2):
        list_new = []
        for i in range(len(list2)):
            list_sum = 0.0
            for j in range(len(list1)):
                list_sum += list1[j] * list2[i]
            list_new.append(list_sum)
        return list_new

    @staticmethod
    def list_each_list_sub(list1, list2):
        list_new = []
        for i in range(len(list1)):
            for j in range(len(list2)):
                list1[i][j] = list1[i][j] - list2[j]
        return list1

    @staticmethod
    def vaule_list_multiply(value, list1):
        list_new = []
        for i in range(len(list1)):
            list_new.append(list1[i] * value)
        return list_new

    @staticmethod
    def value_list_add(value, list1):
        list_new = []
        for i in range(len(list1)):
            list_new.append(list1[i] + value)
        return list_new


class Neuronetz:
    def __init__(self, layers):
        self.layers = layers

    def calc_dalta(self, learning_rate, input):
        last_index = len(self.layers)-1
        for i in range(last_index, -1, -1):
            if i != last_index:
                # schritt 6,7 in der Vorlesung
                s = lambda x, y: x * y
                vs = np.vectorize(s)
                a = np.sum(vs(self.layers[i].ableitung, self.layers[i+1].gewichte), axis=1)
                self.layers[i].delta = MathTools.list_each_list_multiply(
                    a, self.layers[i+1].delta)
                if i != 0:
                    # schritt 8,9 in der Vorlesung
                    b = self.layers[i - 1].func
                else:
                    b = input
            else:
                self.layers[i].delta = MathTools.list_each_list_multiply(self.layers[i].ableitung, self.layers[i].e)
                b = self.layers[i - 1].func
            delta_gewichte = MathTools.vaule_list_multiply((-learning_rate),
                                                               MathTools.list_each_list_multiply(
                                                                   self.layers[i].delta, b))
            self.layers[i].gewichte = MathTools.list_each_list_sub(self.layers[i].gewichte,
                                                                   np.transpose(delta_gewichte))

    def training(self, learning_rate, iteration, label, training_data):
        neuron_length = len(self.layers)
        training_data.append(1)
        for i in range(iteration):
            for k in range(neuron_length):
                if k != 0:
                    self.layers[k].calc_func_ableitung(self.layers[k - 1].func)
                else:
                    self.layers[k].calc_func_ableitung(training_data)
                if k == (neuron_length - 1):
                    self.layers[k].calc_e(label)
            self.calc_dalta(learning_rate, training_data)

    def test(self, test_data):
        last_index = len(self.layers)
        for i in range(last_index):
            if i != 0:
                self.layers[i].calc_func_ableitung(self.layers[i-1].func)
                if i == (last_index - 1):
                    a = np.array(self.layers[i].func)
                    return np.where(np.max(a))
            else:
                self.layers[i].calc_func_ableitung(test_data)


def test(neuronetz):
    testfilename = './datasource/test/zip.test'
    testfile = open(testfilename, 'r')
    print("File for testing is open: ", testfile.name)
    errors = 0
    all = 0
    errorrate = {}
    for i in range(10):
        errorrate[i] = 0
    for line in testfile:
        digit = int(line[:1])
        arr = list(map(float, line[2:].split(' ')))
        arr.append(1)
        foundDigit = neuronetz.test(arr)
        if digit != foundDigit:
            errors += 1
            errorrate[digit] += 1
        all += 1
    print("All tests: ", all)
    print("Count of all errors: ", errors)
    print("Error for every Digit: ", errorrate)


def start():
    trainigfolder = './datasource/training/'
    learning_rate = 0.01
    iteration = 100
    layer1 = Layer(50, 256)
    layer2 = Layer(10, 50)
    neuronetz = Neuronetz(np.array([layer1, layer2]))
    data = Data()
    data.taketrainingdict(trainigfolder)
    for i in range(len(data.traingdata)):      # len(data) : 10
        print(i)
        for j in range(len(data.traingdata[i])):       # len(data[i]) : die Anzahl allen Punkte einer Ziffer
            neuronetz.training(learning_rate, iteration, i, data.traingdata[i][j])
    test(neuronetz)


start()
