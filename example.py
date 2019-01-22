from skmultiflow.lazy.knn import KNN
from skmultiflow.data.sea_generator import SEAGenerator
import numpy as np
from learn_pp import LearnPP

# Setting up the stream
stream = SEAGenerator(1)
stream.prepare_for_use()

# Setting up the Learn++ classifier to work with KNN classifiers
clf = LearnPP(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=30)

# Keeping track of sample count and correct prediction count
sample_count = 0
corrects = 0

m = 200

# Pre training the classifier with 200 samples
X, y = stream.next_sample(m)
clf = clf.partial_fit(X, y, classes=stream.target_values)


for i in range(3):
    X, y = stream.next_sample(m)
    pred = clf.predict(X)
    clf = clf.partial_fit(X, y)
    if pred is not None:
        corrects += np.sum(y == pred)
    sample_count += m
# Displaying the results
print('Learn++ classifier performance: ' + str(corrects / sample_count))
