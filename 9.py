import numpy as np
import matplotlib.pyplot as plt


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
            self.traingdata[digit] = self.normalization(np.array(digitarrs))


    def normalization(self, arr):
        m, n = np.shape(arr)
        #x0 = np.ones(m)
        nor_arr = []
        #label_arr = arr[:, n - 1]
        for i in range(n):
            xi = arr[:, i]
            mean = np.mean(xi)
            var = np.var(xi)
            if (var == 0):
                nor_arr.append(xi)
            else:
                nor_arr.append((xi - mean)/var)
        nor_arr = np.transpose(nor_arr)
        return nor_arr

class Layer:
    def __init__(self, neuronen, input_anzahl, last_layer):  # fuer die esrte Schichte sind es attributen
        # self.gewichte = np.random.rand(neuronen, input_anzahl+1)
        if last_layer:
            self.gewichte = np.random.rand(input_anzahl + 1, neuronen)
            self.func = np.ones(neuronen)     #output auf jeder Schichte
        else:
            self.gewichte = np.random.rand(input_anzahl + 1, neuronen + 1)
            self.func = np.ones(neuronen + 1)   #output auf jeder Schichte
        self.neuronen = neuronen
        self.input_anzahl = input_anzahl
        self.ableitung = []
        self.e = []
        self.delta = []

    def calc_func_ableitung(self, input):   # input ist hier vorherige Ergebnisse
        #  schritt 1,4 in der Vorlesung
        #self.func = MathTools.calc_func(input, self.gewichte)
        #self.ableitung = MathTools.calc_ableitung(self.func)
        self.func = MathTools.sigmoid_vector(np.dot(input, self.gewichte))
        self.ableitung = self.func * (1 - self.func)

    def calc_func(self, input):   # input ist hier vorherige Ergebnisse
        self.func = MathTools.sigmoid_vector(np.dot(input, self.gewichte))

    def calc_e(self, target):      # schritt 3 in der Vorlesung
        targetarr = np.zeros(10)
        targetarr[target] = 1
        self.e = list(map(lambda x: x[0]-x[1], zip(self.func, targetarr)))


class MathTools:
    def __init__(self):
        self = self

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sigmoid_vector(vector):
        for i in range(len(vector)):
            vector[i] = 1 / (1 + np.e ** (-vector[i]))
        return vector

    @staticmethod
    def calc_func(list1, list2):
        list_new = []
        for i in range(len(list2)):
            list_new.append(MathTools.sigmoid(np.dot(list1, list2[i])))
        return list_new

    @staticmethod
    def calc_ableitung(list1):
        list_new = []
        for i in range(len(list1)):
            list_new.append(list1[i]*(1 - list1[i]))
        return list_new

    @staticmethod
    def calc_delta(gewichte, delta, ableitung):
        #list_new = []
        #for i in range(len(list1[1])):
        #    new = 0.0
        #    for j in range(len(list1)-1):
        #        new += list1[j][i] * list2[j]
        #    new *= list3[i]
        #    list_new.append(new)
        return ableitung * np.dot(gewichte, delta)

    @staticmethod
    def calc_delta_gewichte(learning_rate, delta, func):
        list_new = []
        for i in range(len(delta)):
            list1 = []
            for j in range(len(func)):
                list1.append(- learning_rate * delta[i] * func[j])
                print(np.shape(delta[i]))
            list_new.append(list1)
        print(np.shape(list_new))
        return list_new

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
    def update_gewichte(list1, list2):
        list_new = []
        for i in range(len(list1)):
            new = []
            for j in range(len(list2)):
                new.append(list1[i][j] + list2[j])
            list_new.append(new)
        return list_new

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

    @staticmethod
    def value_list_sub(value, list1):
        list_new = []
        for i in range(len(list1)):
            list_new.append(-list1[i] + value)
        return list_new


class Neuronetz:
    def __init__(self, layers):
        self.layers = layers

    def calc_dalta(self, learning_rate, input):
        last_index = len(self.layers)-1
        for i in range(last_index, -1, -1):
            if i != last_index:
                # schritt 6,7 in der Vorlesung
                self.layers[i].delta = self.layers[i].ableitung * np.dot(self.layers[i+1].gewichte, self.layers[i+1].delta)
                # O fuer Schritt 8,9 in der Vorlesung
            else:
                self.layers[i].delta = self.layers[i].ableitung * self.layers[i].e
                #vorherige O
        for i in range(last_index, -1, -1):
            # KOREKTUR
            if i != 0:
                delta_gewichte = np.outer(self.layers[i - 1].func, (learning_rate * self.layers[i].delta))
            else:
                delta_gewichte = np.outer(input, (learning_rate * self.layers[i].delta))
            self.layers[i].gewichte = self.layers[i].gewichte - delta_gewichte

    def training(self, learning_rate, iteration, label, training_data):
        layer_length = len(self.layers)
        training_data = np.append(training_data, 1)   #vector
        for i in range(iteration):
            for k in range(layer_length):
                if k != 0:
                    self.layers[k].calc_func_ableitung(self.layers[k - 1].func)
                else:
                    self.layers[k].calc_func_ableitung(training_data)
                if k == (layer_length - 1):
                    self.layers[k].calc_e(label)
            self.calc_dalta(learning_rate, training_data)

    def test(self, test_data):
        last_index = len(self.layers)
        for i in range(last_index):
            if i != 0:
                self.layers[i].calc_func(self.layers[i-1].func)
                if i == (last_index - 1):
                    a = np.array(self.layers[i].func)
                    return a.argmax()
            else:
                self.layers[i].calc_func(test_data)


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
    learning_rate = 0.5
    iteration = 300
    layer1 = Layer(50, 256, False)
    layer2 = Layer(10, 50, True)
    neuronetz = Neuronetz([layer1, layer2])
    data = Data()
    data.taketrainingdict(trainigfolder)
    for i in range(len(data.traingdata)):      # len(data) : 10
        print(i)
        for j in range(len(data.traingdata[i])):       # len(data[i]) : die Anzahl allen Punkte einer Ziffer
            neuronetz.training(learning_rate, iteration, i, data.traingdata[i][j])
    test(neuronetz)


start()
