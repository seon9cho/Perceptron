import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, max_iter=1000):
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
        self.max_iter = max_iter

    def fit(self, X, y, initial_weights=None, deterministic=False):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.initial_weights = self.initialize_weights(X) if not initial_weights else initial_weights
        
        self.weights = self.initial_weights.copy()
        self.rates = []
        if deterministic:
            for i in range(deterministic):
                self._epoch(X, y)
                self.iteration += 1
        else:
            scale = max(np.abs(X).mean(axis=0)) * (X.shape[1] / 3)
            eps = self.lr * scale
            converged = True
            for i in range(5):
                self._epoch(X, y)
                self.iteration += 1
                self.rates.append(self.miscl_rate)
            while np.linalg.norm(self.deltas) > eps:
                self._epoch(X, y)
                self.iteration += 1
                self.rates.append(self.miscl_rate)
                if self.iteration >= self.max_iter:
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
        self.miscl_rate = 0
        if self.shuffle:
            _X, _y = self._shuffle_data(X, y)
        else:
            _X, _y = X, y
        for x,t in zip(_X, _y):
            pattern = np.append(x, 1)
            output = 0 if np.dot(self.weights, pattern) <= 0 else 1
            self.weights += self.lr * (t[0] - output) * pattern
            self.miscl_rate += int(t[0] == output)
        self.miscl_rate /= len(y)
        self.delta = max(np.abs(self.weights - initial_weights))
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
            output = 0 if np.dot(self.weights, pattern) <= 0 else 1
            pred.append(output)
        return np.array(pred).reshape(-1, 1)

    def initialize_weights(self, X):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """

        return np.zeros(X.shape[1] + 1)

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

def train_test_split(X, y, train_size=0.7, shuffle=True):
    """Split the data into train and test sets
    Args:
        X (array-like): A 2D numpy array with the training data, excluding targets
        y (array-like): A 2D numpy array with the training targets
        train_size (float): Percentage of the data that will be used for training;
            Test size is automatically set to 1 - train_size
        shuffle (bool): Whether to shuffle the dataset or not
    Returns:
        X_train (array-like): A 2D numpy array with the inputs of the training data
        y_train (array-like): A 2D numpy array with the targets of the training data
        X_test (array-like): A 2D numpy array with the inputs of the test data
        y_test (array-like): A 2D numpy array with the targets of the test data
    """
    split = int(train_size * X.shape[0])
    Xy = np.hstack([X, y])
    if shuffle:
        np.random.shuffle(Xy)
    Xy_train = Xy[:split]
    Xy_test = Xy[split:]
    return Xy_train[:,:-1], Xy_train[:,-1].reshape(-1, 1), \
           Xy_test[:,:-1], Xy_test[:,-1].reshape(-1, 1)
    