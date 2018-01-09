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


# def mean(X):
#     return np.mean(X, axis=0)
#
#
# def covariance_matrix(X, mu):
#     num_samples, _ = X.shape
#     X_normalized = X - mu
#     return X_normalized.T.dot(X_normalized) / num_samples
#
#
# def univariate_normal_distribution_pdf(x, mu, sigma):
#     x_normalised = x - mu
#     exponent = -(x_normalised ** 2 / (2 * sigma ** 2))
#     normalisation_term = 1. / np.sqrt(2 * np.pi * sigma ** 2)
#     return normalisation_term * np.e ** exponent


# class FisherDiscriminantClassifier(Classifier):
#
#     def fit(self, X, y):
#         X_pos, X_neg = self._split_binary(X, y)
#         self._find_best_axis(X_pos, X_neg)
#         self._compute_parameters(X_pos, X_neg)
#
#     def _split_binary(self, X, y):
#         labels = np.unique(y)
#         return X[y == labels[1]], X[y == labels[0]]
#
#     def _find_best_axis(self, X_pos, X_neg):
#         mu1 = mean(X_pos)
#         sigma1 = covariance_matrix(X_pos, mu1)
#
#         mu2 = mean(X_neg)
#         sigma2 = covariance_matrix(X_neg, mu2)
#
#         self.u = pinv(sigma1 + sigma2).dot(mu1 - mu2)
#
#     def _compute_parameters(self, X_pos, X_neg):
#         X_pos_transformed = self._transform(X_pos)
#         self.mu1 = mean(X_pos_transformed)
#         self.sigma1 = np.var(X_pos_transformed)
#
#         X_neg_transformed = self._transform(X_neg)
#         self.mu2 = mean(X_neg_transformed)
#         self.sigma2 = np.var(X_neg_transformed)
#
#     def _transform(self, X):
#         return np.matmul(X, self.u)
#
#     def predict(self, X):
#         X_transformed = self._transform(X)
#
#         probs_pos = univariate_normal_distribution_pdf(X_transformed, self.mu1, self.sigma1)
#         probs_neg = univariate_normal_distribution_pdf(X_transformed, self.mu2, self.sigma2)
#
#         results = np.ones(len(X))
#         results[probs_pos < probs_neg] = 0
#         return results

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
            alpha = 0.5 * log((1 - em) / em)
            self.alpha_list.append(alpha)
            one_list = []
            for j in range(data_size):
                one_list.append(X[j])
                if clf_list[i].score(weight[j] * one_list, y[j]) == 0.0:
                    weight[j] = weight[j] * (np.e ** alpha)
                else:
                    weight[j] = weight[j] * (np.e ** (-alpha))
                one_list = []
        self.clf_list = clf_list

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








