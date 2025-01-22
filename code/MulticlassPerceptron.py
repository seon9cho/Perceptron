import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MulticlassPerceptron(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, maxiter=1000):
        """ Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.delta = None
        self.deltas = []
        self.iteration = 0
        self.maxiter = maxiter

    def fit(self, X, y, initial_weights=None, deterministic=False):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.c = int(y.max()) + 1
        self.k = X.shape[1]
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        
        self.weights = self.initial_weights.copy()
        
        if deterministic:
            for i in range(deterministic):
                self._epoch(X, y)
                self.iteration += 1
        else:
            scale = max(np.abs(X).mean(axis=0))
            eps = self.lr * scale
            converged = True
            for i in range(5):
                self._epoch(X, y)
                self.iteration += 1
            while np.linalg.norm(self.deltas) > eps and self.iteration < 10 + 1/self.lr:
                self._epoch(X, y)
                self.iteration += 1
                if self.iteration >= self.maxiter:
                    converged = False
                    break
            if converged:
                print("Classifer converged after {} iterations.".format(self.iteration))
            else:
                print("Maximum number of iterations reached.")
        return self
    
    def _epoch(self, X, y):
        """ 
        Run an epoch through the training dataset X, y
        Args: 
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        """
        initial_weights = self.weights.copy()
        if self.shuffle:
            _X, _y = self._shuffle_data(X, y)
        else:
            _X, _y = X, y
        for x,t in zip(_X, _y):
            t_onehot = np.zeros(self.c)
            t_onehot[int(t[0])] = 1
            pattern = np.append(x, 1)
            output = np.dot(self.weights, pattern).argmax()
            output_onehot = np.zeros(self.c)
            output_onehot[output] = 1
            self.weights += self.lr * (t_onehot - output_onehot).reshape(-1, 1) * pattern
        self.delta = np.abs(self.weights - initial_weights).max()
        if len(self.deltas) < 5:
            self.deltas.append(self.delta)
        else:
            self.deltas.pop(0)
            self.deltas.append(self.delta)

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pred = []
        for x in X:
            pattern = np.append(x, 1)
            output = np.dot(self.weights, pattern).argmax()
            pred.append(output)
        return np.array(pred).reshape(-1, 1)

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """

        return np.zeros((self.c, self.k + 1))

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        pred = self.predict(X)
        return sum(pred==y)[0] / len(y)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        Xy = np.hstack([X, y])
        np.random.shuffle(Xy)
        return Xy[:, :-1], Xy[:, -1].reshape(-1, 1)

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights