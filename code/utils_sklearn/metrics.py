import numpy as np

def MEE(Y,d):
    """
    Calculate the Mean Euclidean Error (MEE) between the predicted values Y and the target values d.

    Parameters:
    Y (numpy.ndarray): Predicted values. Can be 2D array.
    d (numpy.ndarray): Target values. Can be a 2D array.

    Returns:
    float: The mean Euclidean error.
    """
    square_distance = np.sum(np.square(Y - d), axis=1)
    return np.mean(np.sqrt(square_distance))

def MEE_std(Y,d):
    """
    Calculate the standard deviation of the Mean Euclidean Error (MEE) between the predicted values Y and the target values d.

    Parameters:
    Y (numpy.ndarray): Predicted values. Can be a 2D array.
    d (numpy.ndarray): Target values. Can be a 2D array.

    Returns:
    float: The standard deviation of the mean Euclidean error.
    """
    square_distance = np.sum(np.square(Y-d),axis=1)
    return np.std(np.sqrt(square_distance))/np.sqrt(len(square_distance))
    