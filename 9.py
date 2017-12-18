import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, neuronen, input_anzahl):  # fuer die esrte Schichte sind es attributen
        self.gewichte = np.random.rand(neuronen, input_anzahl)
        self.neuronen = neuronen
        self.input_anzahl = input_anzahl
        self.func = []
        self.ableitung = []
        self.e = []
        self.delta = []

    def calc_func_ableitung(self, input):   # input ist hier vorherige Ergebnisse
        input.append(1)
        sigmoid = lambda x: 1 / (1 + np.e ** (-x))
        vf = np.vectorize(sigmoid)
        self.func = vf(input * self.gewichte)    # output auf jeder Schichte       # schritt 1,2 in der Vorlesung
        self.ableitung = self.func * (1 - self.func)    # schritt 4,5 in der Vorlesung

    def calc_e(self, last_output, target):      # schritt 3 in der Vorlesung
        self.e = np.array(last_output) - np.array(target)


class Neuronetz:
    def __init__(self, layers):
        self.layers = layers

    def calc_dalta(self, learning_rate, input):
        last_index = len(self.layers)-1
        for i in range(last_index, -1, -1):
            if i != last_index:
                # schritt 6,7 in der Vorlesung
                self.layers[i].delta = np.dot(np.dot(self.layers[i].ableitung, self.layers[i+1].gewichte), self.layers[i+1].delta)
                if i != 0:
                    # schritt 8,9 in der Vorlesung
                    delta_gewichte = - learning_rate * np.dot(self.layers[i], self.layers[i-1].func)
                else:
                    delta_gewichte = - learning_rate * np.dot(self.layers[i], input.append(1))
            else:
                self.layers[i].delta = np.dot(self.layers[i].ableitung, self.layers[i].e)
                delta_gewichte = - learning_rate * np.dot(self.layers[i], self.layers[i - 1].func)
            self.layers[i].gewichte = self.layers[i].gewichte - delta_gewichte




def start():
    layer1 = Layer(2000, 256 + 1)
    layer2 = Layer(1000, 2000)

start()