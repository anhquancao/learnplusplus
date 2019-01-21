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
clf = LearnPP(base_learner, n_estimators=30)

classes = list(range(0, 10))


def acc(y, y_predict):
    return np.sum(y == y_predict) / len(y)


for i in range(n_subsets):
    print("========================")
    print("Subset", i)
    start = i * 200
    end = (i + 1) * 200
    clf.partial_fit(X[start:end], y[start:end], classes)

    print("Avg Training acc:", clf.total_acc / 30)

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
avg acc: 0.619
Test Avg 0.3347719940709741
Subset 0 COMBINE Training acc 0.865
COMBINE Test acc 0.6664922835469527
========================
Subset 1
avg acc: 0.6173333333333334
Test Avg 0.33095300374923714
Subset 0 COMBINE Training acc 0.91
Subset 1 COMBINE Training acc 0.915
COMBINE Test acc 0.8310227569971227
========================
Subset 2
avg acc: 0.6341666666666664
Test Avg 0.3251198883948033
Subset 0 COMBINE Training acc 0.9
Subset 1 COMBINE Training acc 0.935
Subset 2 COMBINE Training acc 0.935
COMBINE Test acc 0.8849071409887523
========================
Subset 3
avg acc: 0.6726666666666669
Test Avg 0.2903391751678438
Subset 0 COMBINE Training acc 0.915
Subset 1 COMBINE Training acc 0.955
Subset 2 COMBINE Training acc 0.925
Subset 3 COMBINE Training acc 0.95
COMBINE Test acc 0.9000784724038713
========================
Subset 4
avg acc: 0.6666666666666665
Test Avg 0.3224605458191647
Subset 0 COMBINE Training acc 0.925
Subset 1 COMBINE Training acc 0.955
Subset 2 COMBINE Training acc 0.91
Subset 3 COMBINE Training acc 0.96
Subset 4 COMBINE Training acc 0.995
COMBINE Test acc 0.9123724823437092
========================
Subset 5
avg acc: 0.6359999999999999
Test Avg 0.31851948731362806
Subset 0 COMBINE Training acc 0.935
Subset 1 COMBINE Training acc 0.96
Subset 2 COMBINE Training acc 0.94
Subset 3 COMBINE Training acc 0.96
Subset 4 COMBINE Training acc 0.975
Subset 5 COMBINE Training acc 0.93
COMBINE Test acc 0.9278053884383992
"""
