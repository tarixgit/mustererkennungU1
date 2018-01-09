import numpy as np
import pandas as pd
from math import  log
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
#
    def predict(self, X):
        X_attribute = X[:, self.m]
        results = np.ones(len(X))
        for i in range(len(X_attribute)):
            if X_attribute[i] < self.fit_attribute:
                results[i] = 0.0
        return results

#Adaboost algorithm
class Adaboost():
    def fit_ada(self, X, y, clf_list):
        m = len(clf_list)
        self.alpha_list = []
        data_size = len(y)
        weight = np.ones(data_size)
        weight.shape = (data_size, 1)
        for i in range(m):
            clf_list[i].fit(weight * X, y)
            em = 1 - clf_list[i].score(X, y)
            # alpha ist die Klassifikator-Gewichte
            alpha = 0.5 * log((1 - em) / em)
            self.alpha_list.append(alpha)
            one_list = []
            for j in range(data_size):
                one_list.append(X[j])
                # Funktion score arbeitet nur mit der Matrize, deswegen haben wir hier "one_list"
                # hier berechnen wir die die Gewichte fuer Datenpunkt
                if clf_list[i].score(weight[j] * one_list, y[j]) == 0.0:
                    weight[j] = weight[j] * (np.e ** alpha)
                else:
                    weight[j] = weight[j] * (np.e ** (-alpha))
                one_list = []
        self.clf_list = clf_list

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

    # def fit_copy(m):
    #     data = pd.read_csv("datasource/spambase.data", header=None).as_matrix()
    #     X = data[:, :-1]
    #     y = data[:, -1]
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)
    #     clf_list = []
    #     alpha_list = []
    #     data_size = len(y_train)
    #     weight = np.ones(data_size)
    #     weight.shape = (data_size, 1)
    #     for i in range(m):
    #         clf = FisherDiscriminantClassifier()
    #         clf.fit(weight * X_train, y_train)
    #         em = clf.score(X_train, y_train)
    #         alpha = 0.5 * log((1 - em)/em)
    #         alpha_list.append(alpha)
    #         one_list = []
    #         for j in range(data_size):
    #             one_list.append(X_train[j])
    #             if clf.score(weight[j]*one_list, y_train[j]) == 0.0:
    #                 weight[j] = weight[j] * (np.e ** alpha)
    #             else:
    #                 weight[j] = weight[j] * (np.e ** (-alpha))
    #             one_list = []
    #         clf_list.append(clf)
    #     alpha_size = len(alpha_list)
    #     sum_ada = np.zeros(len(y_test))
    #     for i in range(alpha_size):
    #         sum_ada += alpha_list[i] * (clf_list[i].predict(X_test))
    #     for i in range(len(y_test)):
    #         if sum_ada[i] < 0:
    #             sum_ada[i] = 1
    #         else:
    #             sum_ada[i] = 0
    #     print(np.mean(sum_ada == y_test))


def ada_test():
    data = pd.read_csv("datasource/spambase.data", header=None).as_matrix()
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)
    clf_list1 = []
    clf1 = FeatureClassifier()
    clf1.fit_m(X_train, y_train, 55)
    clf_list1.append(clf1)
    clf2 = FeatureClassifier()
    clf2.fit_m(X_train, y_train, 54)
    clf_list1.append(clf2)
    clf3 = FeatureClassifier()
    clf3.fit_m(X_train, y_train, 13)
    clf_list1.append(clf3)
    clf4 = FeatureClassifier()
    clf4.fit_m(X_train, y_train, 52)
    clf_list1.append(clf4)
    clf5 = FeatureClassifier()
    clf5.fit_m(X_train, y_train, 51)
    clf_list1.append(clf5)
    clf6 = FeatureClassifier()
    clf6.fit_m(X_train, y_train, 22)
    clf_list1.append(clf6)
    clf7 = FeatureClassifier()
    clf7.fit_m(X_train, y_train, 25)
    clf_list1.append(clf7)
    clf8 = FeatureClassifier()
    clf8.fit_m(X_train, y_train, 24)
    clf_list1.append(clf8)
    clf9 = FeatureClassifier()
    clf9.fit_m(X_train, y_train, 10)
    clf_list1.append(clf9)
    clf10 = FeatureClassifier()
    clf10.fit_m(X_train, y_train, 16)
    clf_list1.append(clf10)
    score_list = []
    clf_list = []
    for i in range(len(clf_list1)):
        score_list.append(clf_list1[i].score(X_train, y_train))
    for i in np.argsort(score_list[::-1]):
        clf_list.append(clf_list1[i])
    ada = Adaboost()
    ada.fit_ada(X_train, y_train, clf_list)
    ada.predict_ada(X_test, y_test)


ada_test()


# Ich denke, fuer basis classifier koennen wir diese fischer linear classifier nutzen.
# und diese classifier(alle codes daoben) habe ich von der Musterloesung kopiert








