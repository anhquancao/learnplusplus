from sklearn.tree import DecisionTreeClassifier
from skmultiflow.utils import check_random_state
import copy
import numpy as np


class LearnPP:
    """ Learn++ Classifier

    Learn++ is an ensemble learning method introduced by Robi Polikar,
    Lalita Udpa, Satish S. Udpa and Vasant Honavar which is an algorithm
    for incremental training of classifier. The algorithm does not require access to previously
    used data during subsequent incremental learning sessions. At
    the same time, it does not forget previously acquired knowledge.
    Learn++ utilizes ensemble of classifiers by generating multiple
    hypotheses using training data sampled according to carefully
    tailored distributions

    Parameters
    ----------
    base_estimator: StreamModel
        This is the ensemble classifier type, each ensemble classifier is going
        to be a copy of the base_estimator.

    n_estimators: int (default=30)
        The number of classifiers per ensemble

    n_ensembles: int (default=10)
        The number of ensembles to keep.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Raises
    ------
    RuntimeError:
        A RuntimeError is raised if the base_estimator is too weak. In other word,
        it has too low accuracy on the dataset.

        A RuntimeError is raised if the 'classes' parameter is not
        passed in the first partial_fit call, or if they are passed in further
        calls but differ from the initial classes list passed.

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.meta.learn_pp import LearnPP
    >>> from skmultiflow.lazy.knn import KNN
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(1)
    >>> stream.prepare_for_use()
    >>> # Setting up the Learn++ classifier to work with KNN classifiers
    >>> clf = LearnPP(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=30)
    >>> # Keeping track of sample count and correct prediction count
    >>> sample_count = 0
    >>> corrects = 0
    >>> m = 200
    >>> # Pre training the classifier with 200 samples
    >>> X, y = stream.next_sample(m)
    >>> clf = clf.partial_fit(X, y, classes=stream.target_values)
    >>> for i in range(3):
    ...     X, y = stream.next_sample(m)
    ...     pred = clf.predict(X)
    ...     clf = clf.partial_fit(X, y)
    ...     if pred is not None:
    ...         corrects += np.sum(y == pred)
    ...     sample_count += m
    >>>
    >>> # Displaying the results
    >>> print('Learn++ classifier performance: ' + str(corrects / sample_count))
    Learn++ classifier performance: 0.9555

    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=30, n_ensembles=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.ensembles = []
        self.ensemble_weights = []
        self.classes = None
        self.n_ensembles = n_ensembles
        self.random = check_random_state(random_state)

    def partial_fit(self, X, y=None, classes=None):
        """
        partial_fit

        Partially fits the model, based on the X and y matrix.


        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: list
            List of all existing classes. This is an optional parameter, except
            for the first partial_fit call, when it becomes obligatory.

        Raises
        ------
            RuntimeError:
                A RuntimeError is raised if the 'classes' parameter is not
                passed in the first partial_fit call, or if they are passed in further
                calls but differ from the initial classes list passed.

                A RuntimeError is raised if the base_estimator is too weak. In other word,
                it has too low accuracy on the dataset.

        Returns
        _______
        LearnPP
            self

        """
        if self.classes is None:
            if classes is None:
                raise RuntimeError("Should pass the classes in the first partial_fit call")
            else:
                self.classes = classes

        if classes is not None and self.classes is not None:
            if set(classes) == set(self.classes):
                pass
            else:
                raise RuntimeError("The values of classes are different")

        ensemble = [copy.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        normalized_errors = [1.0 for _ in range(self.n_estimators)]

        m = len(X)
        X = np.array(X)
        y = np.array(y)

        Dt = np.ones((m,)) / m

        items_index = np.linspace(0, m - 1, m)
        t = 0
        while t < self.n_estimators:
            print("Generate estimator", t)
            patience = 0

            # Set distribution Dt
            Dt = Dt / np.sum(Dt)

            total_error = 1.0
            while total_error >= 0.5:

                # create training and testing subsets according to Dt
                train_size = int(m / 2)
                test_size = int(m / 2)
                train_items_index = self.get_item(items_index, Dt, train_size)
                test_items_index = self.get_item(items_index, Dt, test_size)

                X_train = X[train_items_index]
                y_train = y[train_items_index]
                X_test = X[test_items_index]
                y_test = y[test_items_index]

                # Train a weak learner
                ensemble[t] = copy.deepcopy(self.base_estimator)
                ensemble[t].fit(X_train, y_train)

                # predict on the data
                y_predict = ensemble[t].predict(X_test)

                total_error = self.compute_error(Dt[test_items_index], y_test, y_predict)

                # print("Error 1" , total_error)
                if total_error < 0.5:
                    # print("Error < 0.5", total_error)
                    norm_error = total_error / (1 - total_error)
                    normalized_errors[t] = norm_error

                    # predict using all hypothesis in the ensemble with majority votes
                    y_predict_composite = self.majority_vote(X, t + 1, ensemble, normalized_errors)

                    total_error = self.compute_error(Dt, y, y_predict_composite)
                    if total_error < 0.5:
                        normalize_composite_error = total_error / (1 - total_error)
                        if t < self.n_estimators - 1:
                            Dt[y_predict_composite == y] = Dt[y_predict_composite == y] * normalize_composite_error

                # print("Error 2", total_error)

                if total_error > 0.5:
                    patience += 1
                if patience > 20:
                    raise RuntimeError("Your base estimator is too weak")
            t += 1

        self.ensembles.append(ensemble)
        self.ensemble_weights.append(normalized_errors)

        if len(self.ensembles) > self.n_ensembles:
            self.ensembles.pop(0)
            self.ensemble_weights.pop(0)

        return self

    def compute_error(self, Dt, y_true, y_predict):
        total_error = np.sum(Dt[y_predict != y_true]) / np.sum(Dt)
        return total_error

    def vote_proba(self, X, t, ensemble, normalized_errors):
        res = []
        for m in range(len(X)):
            votes = np.zeros(len(self.classes))
            for i in range(t):
                h = ensemble[i]

                y_predicts = h.predict(X[m].reshape(1, -1))
                norm_error = normalized_errors[i]
                votes[int(y_predicts[0])] += np.log(1 / (norm_error + 1e-50))

            res.append(votes)
        return res

    def majority_vote(self, X, t, ensemble, normalized_errors):
        res = self.vote_proba(X, t, ensemble, normalized_errors)
        return np.argmax(res, axis=1)

    def predict(self, X):
        """
        predict

        The predict function will use majority votes from all its learners
        with their weights to find the most likely prediction for the sample matrix X.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        votes = np.zeros((len(X), len(self.classes)))
        for i in range(len(self.ensembles)):
            ensemble = self.ensembles[i]
            ensemble_weight = self.ensemble_weights[i]
            votes += np.array(self.vote_proba(X, self.n_estimators, ensemble, ensemble_weight))
        return np.argmax(votes, axis=1)

    def get_item(self, items, items_weights, number_of_items):
        return self.random.choice(items, number_of_items, p=items_weights).astype(np.int32)

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return 'Learn++ Classifier: base_estimator: ' + str(self.base_estimator) + \
               ' - n_estimators: ' + str(self.n_estimators) + " - n_ensembles: " + str(self.n_ensembles)

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError
