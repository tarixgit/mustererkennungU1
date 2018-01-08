import numpy as np
import pandas as pd
from math import  log
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split


class Classifier:
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def confusion_matrix(self, X, y):
        size = len(set(y))
        predicted = self.predict(X)

        results = np.zeros((size, size), dtype=np.int32)

        for pi, yi in zip(predicted, y):
            results[int(pi)][int(yi)] += 1

        return results


def mean(X):
    return np.mean(X, axis=0)


def covariance_matrix(X, mu):
    num_samples, _ = X.shape
    X_normalized = X - mu
    return X_normalized.T.dot(X_normalized) / num_samples


def univariate_normal_distribution_pdf(x, mu, sigma):
    x_normalised = x - mu
    exponent = -(x_normalised ** 2 / (2 * sigma ** 2))
    normalisation_term = 1. / np.sqrt(2 * np.pi * sigma ** 2)
    return normalisation_term * np.e ** exponent


class FisherDiscriminantClassifier(Classifier):

    def fit(self, X, y):
        X_pos, X_neg = self._split_binary(X, y)
        self._find_best_axis(X_pos, X_neg)
        self._compute_parameters(X_pos, X_neg)

    def _split_binary(self, X, y):
        labels = np.unique(y)
        return X[y == labels[1]], X[y == labels[0]]

    def _find_best_axis(self, X_pos, X_neg):
        mu1 = mean(X_pos)
        sigma1 = covariance_matrix(X_pos, mu1)

        mu2 = mean(X_neg)
        sigma2 = covariance_matrix(X_neg, mu2)

        self.u = pinv(sigma1 + sigma2).dot(mu1 - mu2)

    def _compute_parameters(self, X_pos, X_neg):
        X_pos_transformed = self._transform(X_pos)
        self.mu1 = mean(X_pos_transformed)
        self.sigma1 = np.var(X_pos_transformed)

        X_neg_transformed = self._transform(X_neg)
        self.mu2 = mean(X_neg_transformed)
        self.sigma2 = np.var(X_neg_transformed)

    def _transform(self, X):
        return np.matmul(X, self.u)

    def predict(self, X):
        X_transformed = self._transform(X)

        probs_pos = univariate_normal_distribution_pdf(X_transformed, self.mu1, self.sigma1)
        probs_neg = univariate_normal_distribution_pdf(X_transformed, self.mu2, self.sigma2)

        results = np.ones(len(X))
        results[probs_pos < probs_neg] = 0
        return results


def adaboost(m):
    data = pd.read_csv("datasource/spambase.data", header=None).as_matrix()
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)
    clf_list = []
    alpha_list = []
    weight = np.ones(len(y_train))
    for i in range(m):
        clf = FisherDiscriminantClassifier()
        clf.fit(np.transpose(weight) * X_train, y_train)
        em = clf.score(X_train, y_train)
        alpha = 0.5 * log((1 - em)/em)
        alpha_list.append(alpha)
        # achtung
        # achtung
        # achtung
        # hier felht, Gesicht Fehlerrate zu berechnen und Gewicht aktualisieren
        clf_list.append(clf)



# Ich denke, fuer basis classifier koennen wir diese fischer linear classifier nutzen.
# und diese classifier(alle codes daoben) habe ich von der Musterloesung kopiert








