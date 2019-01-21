import pandas as pd
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
        print("Number of estimator", count)

        print("Training Avg", total_acc / count)
        print("Test Avg", total_acc_test / count)
        print("")

    print("COMBINE Test acc", acc(y_test, clf.predict(X_test)))

"""
========================
Subset 0
j 0 COMBINE Training acc 0.935
Training Avg 0.3716666666666667
Test Avg 0.33378672944458965
COMBINE Test acc 0.7802772691603452
========================
Subset 1
j 0 COMBINE Training acc 0.915
Training Avg 0.37116666666666664
Test Avg 0.3497079082744789
j 1 COMBINE Training acc 0.94
Training Avg 0.37675000000000003
Test Avg 0.3497079082744789
COMBINE Test acc 0.8747057284854826
========================
Subset 2
j 0 COMBINE Training acc 0.945
Training Avg 0.35061111111111115
Test Avg 0.33805330310692583
j 1 COMBINE Training acc 0.955
Training Avg 0.3533888888888889
Test Avg 0.33805330310692583
j 2 COMBINE Training acc 0.955
Training Avg 0.3536111111111112
Test Avg 0.33805330310692583
COMBINE Test acc 0.9050483913157207
========================
Subset 3
j 0 COMBINE Training acc 0.95
Training Avg 0.353625
Test Avg 0.3444502572151017
j 1 COMBINE Training acc 0.93
Training Avg 0.35554166666666676
Test Avg 0.3444502572151017
j 2 COMBINE Training acc 0.935
Training Avg 0.3564166666666667
Test Avg 0.3444502572151017
j 3 COMBINE Training acc 0.94
Training Avg 0.3579166666666669
Test Avg 0.3444502572151017
COMBINE Test acc 0.9165576772168454
========================
Subset 4
j 0 COMBINE Training acc 0.95
Training Avg 0.3506000000000001
Test Avg 0.3434911500566746
j 1 COMBINE Training acc 0.935
Training Avg 0.35150000000000015
Test Avg 0.3434911500566746
j 2 COMBINE Training acc 0.94
Training Avg 0.3516666666666666
Test Avg 0.3434911500566746
j 3 COMBINE Training acc 0.94
Training Avg 0.35150000000000015
Test Avg 0.3434911500566746
j 4 COMBINE Training acc 0.94
Training Avg 0.3451666666666667
Test Avg 0.3434911500566746
COMBINE Test acc 0.92545121632226
========================
Subset 5
j 0 COMBINE Training acc 0.955
Training Avg 0.34933333333333333
Test Avg 0.3434359287354318
j 1 COMBINE Training acc 0.945
Training Avg 0.3488611111111114
Test Avg 0.3434359287354318
j 2 COMBINE Training acc 0.945
Training Avg 0.3513888888888889
Test Avg 0.3434359287354318
j 3 COMBINE Training acc 0.92
Training Avg 0.34808333333333347
Test Avg 0.3434359287354318
j 4 COMBINE Training acc 0.925
Training Avg 0.3440277777777779
Test Avg 0.3434359287354318
j 5 COMBINE Training acc 0.945
Training Avg 0.3463055555555555
Test Avg 0.3434359287354318
COMBINE Test acc 0.9322521579911065
"""