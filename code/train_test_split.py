import numpy as np

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