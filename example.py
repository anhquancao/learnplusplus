from sklearn.neural_network import MLPClassifier
from skmultiflow.data.sea_generator import SEAGenerator

# Setting up the stream
from learn_pp import LearnPP

stream = SEAGenerator(1, noise_percentage=6.7)
stream.prepare_for_use()

# Setting up the OzaBagging classifier to work with KNN classifiers
clf = LearnPP(base_estimator=MLPClassifier(), n_estimators=10)

# Keeping track of sample count and correct prediction count
sample_count = 0
corrects = 0

# Pre training the classifier with 200 samples
X, y = stream.next_sample(200)
clf = clf.partial_fit(X, y, classes=stream.target_values)
for i in range(2000):
    X, y = stream.next_sample()
    pred = clf.predict(X)
    clf = clf.partial_fit(X, y)
    if pred is not None:
        if y[0] == pred[0]:
            corrects += 1
    sample_count += 1

# Displaying the results
print(str(sample_count) + ' samples analyzed.')
print('Learn++ classifier performance: ' + str(corrects / sample_count))