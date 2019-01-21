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

base_learner = MLPClassifier(hidden_layer_sizes=(100,), tol=0.0001, max_iter=500)
clf = LearnPP(base_learner, n_estimators=30)

classes = [0, 1, 2, 3]


def acc(y, y_predict):
    return np.sum(y == y_predict) / len(y)


for i in range(n_subsets):
    print("========================")
    print("Subset", i)
    start = i * m
    end = (i + 1) * m
    clf.partial_fit(X[start:end], y[start:end], classes)

    for j in range(i + 1):

        s_i = j * m
        e_i = (j + 1) * m

        X_t = X[s_i: e_i]
        y_t = y[s_i: e_i]

        print(s_i, e_i)

        y_predict = clf.predict(X_t)
        print("j", j, "COMBINE Training acc", acc(y_t, y_predict))

        total_acc = 0
        total_acc_test = 0
        count = 0
        for ensemble in clf.ensembles:
            for h in ensemble:
                y_pred = h.predict(X_t)
                total_acc += acc(y_t, y_pred)

                y_test_pred = h.predict(X_test)
                total_acc_test += acc(y_test, y_test_pred)

                count += 1

        print("Number of estimator", count)

        print("Training Avg", total_acc / count)
        print("Test Avg", total_acc_test / count)

    print("COMBINE Test acc", acc(y_test, clf.predict(X_test)))
