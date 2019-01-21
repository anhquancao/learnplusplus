import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from learn_pp import LearnPP
import numpy as np

data = pd.read_csv("data/xaa.dat.txt", delimiter=' ', header=None).sample(frac=1).reset_index(drop=True)
y = data.iloc[:, 18].values
X = data.iloc[:, :18].values

std = np.std(X, axis=0)
mean = np.mean(X, axis=0)
X = (X - mean) / std

y[y == 'opel'] = 0
y[y == 'van'] = 1
y[y == 'bus'] = 2
y[y == 'saab'] = 3

y = y.astype(np.float32)

m = 210
n_subsets = 3

X_test = X[n_subsets * m:]
y_test = y[n_subsets * m:]

base_learner = MLPClassifier(hidden_layer_sizes=(100,), tol=1e-3, max_iter=500)
clf = LearnPP(base_learner, n_estimators=30, random_state=22)

classes = [0, 1, 2, 3]


def acc(y, y_predict):
    return np.sum(y == y_predict) / len(y)


for i in range(n_subsets):
    print("========================")
    print("Subset", i)
    start = i * 200
    end = (i + 1) * 200
    clf.partial_fit(X[start:end], y[start:end], classes)

    total_acc = 0
    for h in clf.ensembles[i]:
        y_pred = h.predict(X[start:end])
        total_acc += acc(y[start:end], y_pred)
    print("Avg Training acc:", total_acc / 30)

    total_acc = 0
    total_acc_test = 0
    count = 0
    for h in clf.ensembles[i]:
        y_test_pred = h.predict(X_test)
        total_acc_test += acc(y_test, y_test_pred)

    print("Test Avg", total_acc_test / 30)

    for j in range(i + 1):
        X_t = X[j * 200: (j + 1) * 200]
        y_t = y[j * 200: (j + 1) * 200]
        y_predict = clf.predict(X_t)
        print("Subset", j, "COMBINE Training acc", acc(y_t, y_predict))

    print("COMBINE Test acc", acc(y_test, clf.predict(X_test)))

"""
========================
Subset 0
Avg Training acc: 0.7881666666666666
Test Avg 0.7030864197530863
Subset 0 COMBINE Training acc 0.905
COMBINE Test acc 0.7870370370370371
========================
Subset 1
Avg Training acc: 0.8171666666666665
Test Avg 0.7276234567901233
Subset 0 COMBINE Training acc 0.855
Subset 1 COMBINE Training acc 0.855
COMBINE Test acc 0.7916666666666666
========================
Subset 2
Avg Training acc: 0.8146666666666669
Test Avg 0.7266975308641974
Subset 0 COMBINE Training acc 0.82
Subset 1 COMBINE Training acc 0.825
Subset 2 COMBINE Training acc 0.845
COMBINE Test acc 0.8148148148148148
"""
