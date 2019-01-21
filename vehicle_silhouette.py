import pandas as pd
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

base_learner = DecisionTreeClassifier(max_depth=3)
clf = LearnPP(base_learner, n_estimators=30, random_state=5)

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

        X_t = X[j * m: (j + 1) * m]
        y_t = y[j * m: (j + 1) * m]
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

        print("Training Avg", total_acc / count)
        print("Test Avg", total_acc_test / count)
        print("")

    print("COMBINE Test acc", acc(y_test, clf.predict(X_test)))
