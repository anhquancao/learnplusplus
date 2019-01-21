import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from learn_pp import LearnPP
import numpy as np
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("data/optdigits.tra", header=None).sample(frac=1).reset_index(drop=True)
y = data.iloc[:, 64].values
X = data.iloc[:, :64].values

test_data = pd.read_csv("data/optdigits.tes.txt", header=None)
y_test = data.iloc[:, 64].values
X_test = data.iloc[:, :64].values

m = 200
n_subsets = 6
# base_learner = DecisionTreeClassifier(max_depth=3)
base_learner = MLPClassifier(hidden_layer_sizes=(30,), tol=0.1)
clf = LearnPP(base_learner, n_estimators=30, random_state=22)

classes = list(range(0, 10))


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
Avg Training acc: 0.5095
Test Avg 0.4359490801290436
Subset 0 COMBINE Training acc 0.915
COMBINE Test acc 0.8030342662830238
========================
Subset 1
Avg Training acc: 0.5053333333333334
Test Avg 0.42697706861975776
Subset 0 COMBINE Training acc 0.935
Subset 1 COMBINE Training acc 0.895
COMBINE Test acc 0.8524718807219461
========================
Subset 2
Avg Training acc: 0.5275000000000001
Test Avg 0.44041328799372226
Subset 0 COMBINE Training acc 0.905
Subset 1 COMBINE Training acc 0.915
Subset 2 COMBINE Training acc 0.925
COMBINE Test acc 0.8671200627779231
========================
Subset 3
Avg Training acc: 0.5383333333333333
Test Avg 0.4277617926584707
Subset 0 COMBINE Training acc 0.905
Subset 1 COMBINE Training acc 0.92
Subset 2 COMBINE Training acc 0.93
Subset 3 COMBINE Training acc 0.945
COMBINE Test acc 0.8992937483651583
========================
Subset 4
Avg Training acc: 0.5263333333333333
Test Avg 0.4366378934519138
Subset 0 COMBINE Training acc 0.91
Subset 1 COMBINE Training acc 0.9
Subset 2 COMBINE Training acc 0.925
Subset 3 COMBINE Training acc 0.94
Subset 4 COMBINE Training acc 0.95
COMBINE Test acc 0.9034789432382946
========================
Subset 5
Avg Training acc: 0.5153333333333332
Test Avg 0.4313453657685937
Subset 0 COMBINE Training acc 0.92
Subset 1 COMBINE Training acc 0.925
Subset 2 COMBINE Training acc 0.93
Subset 3 COMBINE Training acc 0.945
Subset 4 COMBINE Training acc 0.94
Subset 5 COMBINE Training acc 0.925
COMBINE Test acc 0.9176039759351294
"""
