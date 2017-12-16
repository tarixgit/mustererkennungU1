import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, neuronen, input_anzahl):  ### fuer die esrte Schichte sind es attributen
        self.gewichte = np.random.rand(neuronen, input_anzahl)
        self.neuronen = neuronen
        self.input_anzahl = input_anzahl
        self.func = []
        self.ableitung = []

    def calc_func_anleitung(self, input):   ###input ist hier vorherige Ergebnisse
        sigmoid = lambda x: 1 / (1 + e ** (-x))
        vf = np.vectorize(sigmoid)
        self.func = vf(input * self.gewichte)    #output auf jeder Schichte
        self.ableitung = self.func * (1 - self.func)



def start():
    layer1 = Layer(2000, 256 + 1)
    layer2 = Layer(1000, 2000)

start()