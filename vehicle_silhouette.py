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
Avg Training acc: 0.7886666666666665
Test Avg 0.7141975308641976
Subset 0 COMBINE Training acc 0.915
COMBINE Test acc 0.8194444444444444
========================
Subset 1
Avg Training acc: 0.8509999999999999
Test Avg 0.7132716049382715
Subset 0 COMBINE Training acc 0.795
Subset 1 COMBINE Training acc 0.895
COMBINE Test acc 0.7731481481481481
========================
Subset 2
Avg Training acc: 0.7635000000000001
Test Avg 0.7012345679012345
Subset 0 COMBINE Training acc 0.775
Subset 1 COMBINE Training acc 0.905
Subset 2 COMBINE Training acc 0.775
COMBINE Test acc 0.8148148148148148
"""
