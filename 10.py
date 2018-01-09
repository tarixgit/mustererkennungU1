import numpy as np
import pandas as pd
import copy
from math import log
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split


class Classifier:
# hier wird gerechnet, wie hoch ist die Genauigkeit
# predictions - das ist das Ergebnis vom Klassifikator
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

# naechste Funktion wird nihct benutzt
    def confusion_matrix(self, X, y):
        size = len(set(y))
        predicted = self.predict(X)

        results = np.zeros((size, size), dtype=np.int32)

        for pi, yi in zip(predicted, y):
            results[int(pi)][int(yi)] += 1

        return results

# Das ist unsere Bassisklassifikator, dass die Klasse Classifier veerbert
# es wurde zuerst gedacht, mehr komplexer Klassifkator zu schreiben
# es wurde gesagt, dass es reicht Klassifikator zuschreieben, der mehr als 50 % ausweist
# Deswegen funktioniert unserer Klassifikator aehnlich wie Linear Distribution(Regression)
# Aber nur eingesetzt auf bestimmten Feature(Merkmale)
# Deswegen erstellen wir spaeter Klassifikatoren auf dem Basis von "FeatureClassifier" mit verschiedenen Merkmalen,
# die am besten passen, um Spam-NichtSpam zu unterscheieden
# Merkmale speichern wir hier unter self.m
# self.fit_attribute - Mitte auf einer Achse(auf ausgewaehltem Attribut), die unsere Menge von allen Vektoren von auf Spam-NichSpam trennt
class FeatureClassifier(Classifier):
    def fit_m(self, X, y, m):
        self.fit_attribute = np.mean(X[:, m])
        self.m = m

    def fit(self, X, y):
        self.fit_attribute = np.mean(X[:, self.m])

    def predict(self, X):
        X_attribute = X[:, self.m]
        results = np.ones(len(X))
        for i in range(len(X_attribute)):
            if X_attribute[i] < self.fit_attribute:
                results[i] = 0.0
        return results

#Adaboost algorithm
class Adaboost():
    def fit_ada(self, X, y, clf_list, m):
        self.alpha_list = []
        clf_list_new = []
        data_size = len(y)
        weight = np.ones(data_size)
        weight.shape = (data_size, 1)
        for i in range(m):
            we_list = []
            w_list = []
            for j in range(len(clf_list)):
                clf_list[j].fit(weight * X, y)
                one_list = []
                we = 0.0
                w = 0.0
                for k in range(data_size):
                    one_list.append(X[k])
                    # wi = np.e ** (-y[k] * self.predict_ada(one_list, y[k]))
                    wi = weight[k]
                    w += wi
                    if clf_list[j].score(weight[k] * one_list, y[k]) == 0.0:
                        we += wi
                    one_list = []
                we_list.append(we)
                w_list.append(w)
            we = min(we_list)
            a = we_list.index(we)
            clf_new = copy.deepcopy(clf_list[a])
            clf_list_new.append(clf_new)
            em = we/w
            alpha = 0.5 * log((1 - em) / em)
            self.alpha_list.append(alpha)
            for p in range(data_size):
                one_list.append(X[p])
                if clf_list[a].score(weight[p] * one_list, y[p]) == 0.0:
                    weight[p] = weight[p] * (np.e ** alpha)
                else:
                    weight[p] = weight[p] * (np.e ** (-alpha))
                one_list = []
        self.clf_list = clf_list_new

    # hier ist schon das Testen
    def predict_ada(self, X, y):
        alpha_size = len(self.alpha_list)
        sum_ada = np.zeros(len(y))
        for i in range(alpha_size):
            sum_ada += self.alpha_list[i] * (self.clf_list[i].predict(X))
            print(i + self.clf_list[i].score(X, y))
        for i in range(len(y)):
            if sum_ada[i] > 0:
                sum_ada[i] = 1
            else:
                sum_ada[i] = 0
        print(np.mean(sum_ada == y))


def ada_test(m):
    data = pd.read_csv("datasource/spambase.data", header=None).as_matrix()
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)
    clf_list1 = []
    clf1 = FeatureClassifier()
    clf1.fit_m(X_train, y_train, 55)
    clf_list1.append(clf1)
    clf2 = FeatureClassifier()
    clf2.fit_m(X_train, y_train, 23)
    clf_list1.append(clf2)
    clf3 = FeatureClassifier()
    clf3.fit_m(X_train, y_train, 38)
    clf_list1.append(clf3)
    clf4 = FeatureClassifier()
    clf4.fit_m(X_train, y_train, 29)
    clf_list1.append(clf4)
    clf5 = FeatureClassifier()
    clf5.fit_m(X_train, y_train, 45)
    clf_list1.append(clf5)
    clf6 = FeatureClassifier()
    clf6.fit_m(X_train, y_train, 4)
    clf_list1.append(clf6)
    ada = Adaboost()
    ada.fit_ada(X_train, y_train, clf_list1, m)
    ada.predict_ada(X_test, y_test)


ada_test(20)









