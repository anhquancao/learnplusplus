import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from learn_pp import LearnPP
import numpy as np
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("optdigits.tra", header=None).sample(frac=1).reset_index(drop=True)
y = data.iloc[:, 64].values
X = data.iloc[:, :64].values

test_data = pd.read_csv("optdigits.tes.txt", header=None)
y_test = data.iloc[:, 64].values
X_test = data.iloc[:, :64].values

m = 200
n_subsets = 6
# base_learner = DecisionTreeClassifier(max_depth=4)
base_learner = MLPClassifier(hidden_layer_sizes=(30), tol=0.1)
clf = LearnPP(base_learner, n_estimators=30, random_state=5)

classes = list(range(0, 10))


def acc(y, y_predict):
    return np.sum(y == y_predict) / len(y)


for i in range(n_subsets):
    print("========================")
    print("Subset", i)
    start = i * 200
    end = (i + 1) * 200
    clf.partial_fit(X[start:end], y[start:end], classes)

    for j in range(i + 1):

        X_t = X[j * 200: (j + 1) * 200]
        y_t = y[j * 200: (j + 1) * 200]
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
